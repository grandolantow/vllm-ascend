"""Local TP coordination for the GLM-5.3 MK worker hook."""

from array import array
from contextlib import contextmanager
import fcntl
import mmap
import os
from pathlib import Path
import select
import socket
from struct import Struct
from time import monotonic, sleep


_CREDENTIALS = Struct("3i")
_SETUP_TIMEOUT = 30
_ROUND_TIMEOUT = 180
_PUBLICATION_LOCK_TIMEOUT = 0.25


def _receive(peer, size):
    payload = peer.recv(size + 1)
    if len(payload) != size:
        raise RuntimeError(f"TP control received {len(payload)} bytes, expected {size}")
    return payload


def _send(peer, payload):
    if peer.send(payload) != len(payload):
        raise RuntimeError("TP control sent a partial packet")


class LocalTPControl:
    """One ordered stream of fixed-size collectives in a Linux net namespace."""

    def __init__(self, rank, size, payload_size):
        self.rank, self.size, self.payload_size = rank, size, payload_size
        self.peers = {}
        self.result = None
        self.lock_fd = None
        self.events = []
        self.pollers = []
        self.peer_poll = None
        self.generation = 0

    @classmethod
    def connect(cls, gate, payload_size):
        # Keep initialization on Gloo so all workers see any setup failure.
        try:
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            net = os.stat("/proc/self/ns/net")
            pid = os.stat("/proc/self/ns/pid")
            location = (boot, net.st_dev, net.st_ino, pid.st_dev, pid.st_ino, os.geteuid())
        except (AttributeError, OSError):
            location = None
        workers = gate.gather("local_control_location", (location, os.getpid()))
        if location is None or any(peer_location != location for peer_location, _ in workers):
            raise RuntimeError("local TP control requires one Linux network/PID namespace and user")

        def check_peer(peer, rank):
            credentials = peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _CREDENTIALS.size)
            pid, uid, _ = _CREDENTIALS.unpack(credentials)
            if pid != workers[rank][1] or uid != location[-1]:
                raise RuntimeError("local TP control peer credentials mismatch")

        control = cls(gate.rank, gate.size, payload_size)
        # A leading byte records an abort that occurred before publication.
        byte_count = 1 + gate.size * payload_size
        listener, endpoint, error, transfer_fd = None, None, None, None
        if gate.rank == 0:
            try:
                transfer_fd = os.memfd_create("glm53-mk-tp-control", os.MFD_CLOEXEC)
                os.ftruncate(transfer_fd, byte_count)
                control.lock_fd = os.open(f"/proc/self/fd/{transfer_fd}", os.O_RDWR | os.O_CLOEXEC)
                control.result = mmap.mmap(control.lock_fd, byte_count, access=mmap.ACCESS_WRITE)
                for _ in range(2):
                    control.events.append(os.eventfd(0, os.EFD_CLOEXEC | os.EFD_SEMAPHORE | os.EFD_NONBLOCK))
                endpoint = b"\0glm53-mk-tp-" + os.urandom(16).hex().encode("ascii")
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                listener.settimeout(_SETUP_TIMEOUT)
                listener.bind(endpoint)
                listener.listen(gate.size - 1)
            except (AttributeError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"

        try:
            bound = gate.gather("local_control_bound", (endpoint, error))
            if bound[0][1] or any(peer != (None, None) for peer in bound[1:]):
                raise RuntimeError(f"local TP control bind failed: {bound}")
            try:
                if gate.rank == 0:
                    control.peer_poll = select.poll()
                    for _ in range(1, gate.size):
                        peer, _ = listener.accept()
                        try:
                            peer.settimeout(_SETUP_TIMEOUT)
                            rank = _receive(peer, 1)[0]
                            if rank == 0 or rank >= gate.size or rank in control.peers:
                                raise RuntimeError("invalid local TP control peer rank")
                            check_peer(peer, rank)
                            rights = (socket.SOL_SOCKET, socket.SCM_RIGHTS, array("i", [transfer_fd, *control.events]))
                            if peer.sendmsg([b"\0"], [rights]) != 1:
                                raise RuntimeError("local TP control descriptor transfer failed")
                            control.peers[rank] = peer
                            control.peer_poll.register(peer, 0)
                        except Exception:
                            peer.close()
                            raise
                else:
                    peer = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                    control.peers[0] = peer
                    peer.settimeout(_SETUP_TIMEOUT)
                    peer.connect(bound[0][0])
                    check_peer(peer, 0)
                    _send(peer, bytes([gate.rank]))
                    control._receive_descriptors(peer, byte_count)
                    for event in control.events:
                        poller = select.poll()
                        poller.register(peer, select.POLLIN)
                        poller.register(event, select.POLLIN)
                        control.pollers.append(poller)
                for peer in control.peers.values():
                    peer.settimeout(_ROUND_TIMEOUT)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            gate.check("local_control_connected", error)
            return control
        except Exception:
            control.close()
            raise
        finally:
            if listener is not None:
                listener.close()
            if transfer_fd is not None:
                os.close(transfer_fd)

    def _receive_descriptors(self, peer, byte_count):
        fds = array("i")
        payload, messages, flags, _ = peer.recvmsg(1, socket.CMSG_SPACE(3 * fds.itemsize), socket.MSG_CMSG_CLOEXEC)
        for level, kind, data in messages:
            if (level, kind) == (socket.SOL_SOCKET, socket.SCM_RIGHTS):
                fds.frombytes(data[:len(data) - len(data) % fds.itemsize])
        try:
            if payload != b"\0" or flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC) or len(fds) != 3:
                raise RuntimeError("invalid local TP control descriptor packet")
            # SCM_RIGHTS duplicates the sender's open file description. Reopen
            # it so flock treats every worker as an independent locker.
            self.lock_fd = os.open(f"/proc/self/fd/{fds[0]}", os.O_RDWR | os.O_CLOEXEC)
            self.result = mmap.mmap(self.lock_fd, byte_count, access=mmap.ACCESS_WRITE)
            self.events.extend(fds[1:])
            del fds[1:]
        finally:
            for fd in fds:
                os.close(fd)

    @contextmanager
    def _publication_lock(self):
        deadline = monotonic() + _PUBLICATION_LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise RuntimeError("local TP control publication lock timed out") from exc
                sleep(min(0.001, remaining))
        try:
            yield
        finally:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)

    def _collect(self, payload):
        deadline = monotonic() + _ROUND_TIMEOUT
        rows = [payload]
        for rank in range(1, self.size):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise RuntimeError("local TP control collection timed out")
            peer = self.peers[rank]
            peer.settimeout(remaining)
            rows.append(_receive(peer, self.payload_size))
        if self.peer_poll.poll(0):
            raise RuntimeError("local TP control peer disconnected before publication")
        return b"".join(rows)

    def all_gather(self, payload):
        if not self.peers:
            raise RuntimeError("local TP control is closed")
        if len(payload) != self.payload_size:
            raise RuntimeError("unexpected local TP control payload size")
        # A fast leaf can enter the next gather before a slow leaf takes its
        # notification. Alternate counters so it cannot drain the slow leaf's.
        bank = self.generation % 2
        event = self.events[bank]
        try:
            if self.rank != 0:
                peer = self.peers[0]
                _send(peer, payload)
                poller = self.pollers[bank]
                if not dict(poller.poll(int(_ROUND_TIMEOUT * 1000))).get(event, 0) & select.POLLIN:
                    with self._publication_lock():
                        if not dict(poller.poll(0)).get(event, 0) & select.POLLIN:
                            self.result[0] = 1
                            raise RuntimeError("local TP control round was not published")
                if os.eventfd_read(event) != 1:
                    raise RuntimeError("invalid local TP control notification")
                rows = self.result[1:]
            else:
                rows = self._collect(payload)
                # Publish to every leaf together. A leaf reads this buffer before
                # sending its next packet, so the next full gather can reuse it.
                with self._publication_lock():
                    if self.result[0]:
                        raise RuntimeError("local TP control peer aborted before publication")
                    self.result[1:] = rows
                    os.eventfd_write(event, self.size - 1)
        except (OSError, RuntimeError) as exc:
            self.close()
            raise RuntimeError(f"local TP control failed on rank {self.rank}") from exc
        self.generation += 1
        return rows

    def close(self):
        self.pollers.clear()
        self.peer_poll = None
        for peer in self.peers.values():
            peer.close()
        self.peers.clear()
        if self.result is not None:
            self.result.close()
            self.result = None
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
        for event in self.events:
            os.close(event)
        self.events.clear()
