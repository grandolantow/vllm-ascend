"""Own a single eager native metadata build on a retained MK decode round."""


def release_after_join(executor, join):
    if executor is not None and executor.submission_in_flight:
        join(executor.stream)
        executor.release()


class DeferredNativeMetadata:
    def __init__(self, runner, mode, build, join):
        self.runner = runner
        self.mode = mode
        self.build = build
        self.join = join
        self.executor = None
        self.result = None
        self.attempted = False

    def materialize(self, context):
        if self.result is not None:
            return self.result
        if self.attempted:
            raise RuntimeError("A failed native metadata build cannot be retried in this round")
        runner = self.runner
        executor = runner.device_metadata_executor
        if executor is not None and executor.submission_in_flight:
            raise RuntimeError("Previous native metadata submission is still active")
        self.attempted = True
        try:
            with runner.synchronize_input_prep():
                metadata, draft = self.build()
            if not isinstance(metadata, dict) or draft is not None:
                raise RuntimeError("Expected a native, non-speculative eager metadata dictionary")
            self.executor = runner._prepare_device_metadata_for_forward(self.mode)
            if self.executor is not None:
                # The retained hook may have read context.device_metadata_executor
                # before asking for this fallback. Join before handing out tensors.
                self.join(self.executor.stream)
            context.attn_metadata = metadata
            context.device_metadata_executor = self.executor
            self.result = (metadata, draft)
            return self.result
        except BaseException:
            # submit() marks the executor active before running producer callbacks.
            release_after_join(executor, self.join)
            raise

    def finish(self):
        # Native forward normally releases its submission itself. A late MK
        # fallback instead needs the consumer-side fence after the bridge reads.
        if self.executor is not None and self.executor.submission_in_flight:
            self.executor.release()
