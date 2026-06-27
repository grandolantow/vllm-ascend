import json
import os
import random
import string
import subprocess
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_VERSION"] = "0.20.2"
# os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "14,15"
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
import torch
import torch_npu
from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoTokenizer


from vllm import LLM, SamplingParams, AsyncLLMEngine
from vllm.config import CompilationConfig

# model_path="/mnt/share/z00911889/data/model_from_hf/deepseek-mtp-test/"
# model_path="/mnt/nfs/z00911889/data/model_from_hf/deepseek_mtp_main_random_bf16/"
# model_path="/mnt/share/weights/DeepSeek-R1-0528_w8a8_mix_mtp"
# model_path="/home/weights/DeepSeek-R1-0528_w8a8_mix_mtp"
# model_path="/home/z00911889/data/model_from_hf/DeepSeek-R1-0528_w8a8_mix_mtp"
# model_path="/mnt/weight/DeepSeek-V3.2-Exp-W8A8"
model_path="/mnt/weight/GLM-5-w4a8"
# model_path="/mnt/share/weights/DeepSeek-V2-Lite/"
# model_path="/home/weights/DeepSeek-V2-Lite-Chat"
# model_path="/home/weights/DeepSeek-V2-Lite-W8A8"
# model_path="/home/weights/DeepSeek-V3.2-Exp-W8A8"

def generate_prompts_auto(input_len, batchsize):
    prompts = [" ".join([f"{random.choice(string.ascii_letters)}" for _ in range(input_len)]) for _ in range(batchsize)]
    # print(prompts)
    return prompts

prompts = [
    # "The capital of France is",
    # "Hello, my name is Tom, I am",
    # "The president of United States is",
    # "The capital of France is Paris. It is one of the most famous and visited cities in the world. Paris is",
    # "Hello, my name is Tom, I am a student at the University of Michigan. I am",
    # "The president of United States is the head of state and head of government of the United States. President is",
    # "It is raining outside",
    # "Tom is running on the playground",
    # "The difference between cat and dog",
    # "AI future is? What do you think about it?",
    # "AI future is? What do you think about it? Can you give me some information?",
    # "AI future is? What do you think about it?"
    # "I am",
    # "You",
    # "I am a",
    # "某小区有 3 栋楼，每栋 6 层，每层 2 户。1 号楼 1 层住户姓王, 2 层姓李; 2 号楼 3 层东户姓张，西户与 3 号楼 4 层西户同姓; 3 号楼 5 层东户和 1 号楼 4 层西户都姓赵。已知所有住户姓氏不重复，共 12 个姓，问 2 号楼 6 层西户可能姓什么？",
    "Large language models have demonstrated remarkable linguistic capabilities through unsupervised next token prediction trained on massive nature language corpora, paving the way for the pursuit of artificial general intelligence. Fueled by readily accessible web-scale nature language resources, model parameters have grown exponentially from millions to trillions. Reinforcement learning (RL) has become a pivotal technology in the post-training phase of large language models (LLMs). Traditional task-colocated RL frameworks suffer from significant scalability bottlenecks, while task-separated RL frameworks face challenges in complex dataflows and the corresponding resource idling and workload imbalance. Moreover, most existing frameworks are tightly coupled with LLM training or inference engines, making it difficult to support custom-designed engines. To address these challenges, we propose AsyncFlow, an asynchronous streaming RL framework for efficient post-training. Specifically, we introduce a distributed data storage and transfer module that provides a unified data management and fine-grained scheduling capability in a fully streamed manner. This architecture inherently facilitates automated pipeline overlapping among RL tasks and dynamic load balancing. Moreover, we propose a producer-consumer-based asynchronous workflow engineered to minimize computational idleness by strategically deferring parameter update process within staleness thresholds. Finally, the core capability of AsynFlow is architecturally decoupled from underlying training and inference engines and encapsulated by service-oriented user interfaces, offering a modular and customizable user experience. Extensive experiments demonstrate",
    # "Scarlett O'Hara was not beautiful, but men seldom realized it when caught by her charm as the Tarleton twins were.  In her face were too sharply blended the delicate features of her mother, a Coast aristocrat of French descent, and the heavy ones of her florid Irish father.  But it was an arresting face, pointed of chin, square of jaw.  Her eyes were pale green without a touch of hazel, starred with bristly black lashes and slightly tilted at the ends. Above them, her thick black brows slanted upward, cutting a startling oblique line in her magnolia-white skin--that skin so prized by Southern women and so carefully guarded with bonnets, veils and mittens against hot Georgia suns. Seated with Stuart and Brent Tarleton in the cool shade of the porch of Tara, her father's plantation, that bright April afternoon of 1861, she made a pretty picture.  Her new green flowered-muslin dress spread its twelve yards of billowing material over her hoops and exactly matched the flat-heeled green morocco slippers her father had recently brought her from Atlanta. The dress set off to perfection the seventeen-inch waist, the smallest in three counties, and the tightly fitting basque showed breasts well matured for her sixteen years.  But for all the modesty of her spreading skirts, the demureness of hair netted smoothly into a chignon and the quietness of small white hands folded in her lap, her true self was poorly concealed. The green eyes in the carefully sweet face were turbulent, willful, lusty with life, distinctly at variance with her decorous demeanor. Her", # 2.x blocks
    # "Scarlett O'Hara was not beautiful, but men seldom realized it when caught by her charm as the Tarleton twins were.  In her face were too sharply blended the delicate features of her mother, a Coast aristocrat of French descent, and the heavy ones of her florid Irish father.  But it was an arresting face, pointed of chin, square of jaw.  Her eyes were pale green without a touch of hazel, starred with bristly black lashes and slightly tilted at the ends. Above them, her thick black brows slanted upward, cutting a startling oblique line in her magnolia-white skin--that skin so prized by Southern women and so carefully guarded with bonnets, veils and mittens against hot Georgia suns. Seated with Stuart and Brent Tarleton in the cool shade of the porch of Tara, her father's plantation, that bright April afternoon of 1861, she made a pretty picture.  Her new green flowered-muslin dress spread its twelve yards of billowing material over her hoops and exactly matched the flat-heeled green morocco slippers her father had recently brought her from Atlanta. The dress set off to perfection the seventeen-inch waist, the smallest in three counties, and the tightly", # 2 blocks - 1 token
    # "Scarlett O'Hara was not beautiful, but men seldom realized it when caught by her charm as the Tarleton twins were.  In her face were too sharply blended the delicate features of her mother, a Coast aristocrat of French descent, and the heavy ones of her florid Irish father.  But it was an arresting face, pointed of chin, square of jaw.  Her eyes were pale green without a touch of hazel, starred with bristly black lashes and slightly tilted at the ends. Above them, her thick black brows slanted upward, cutting a startling oblique line in her magnolia-white skin--that skin so prized by Southern women and so carefully guarded with bonnets, veils and mittens against hot Georgia suns. Seated with Stuart and Brent Tarleton in the cool shade of the porch of Tara, her father's plantation, that bright April afternoon of 1861, she made a pretty picture.  Her new green flowered-muslin dress spread its ", # 1.x blocks
    # "Scarlett O'Hara was not beautiful, but men seldom realized it when caught by her charm as the Tarleton twins were.  In her face were too sharply blended the delicate features of her mother, a Coast aristocrat of French descent, and the heavy ones of her florid Irish father.  But it was an arresting face, pointed of chin, square of jaw.  Her eyes were pale green without a touch of hazel, starred with bristly black lashes and slightly tilted at the ends. Above them, her thick black brows slanted upward, cutting a startling oblique line in her magnolia-white skin", # 1 block - 1 token
    # "Two track teams are competing against each other in a 4 by 400 meter relay; a race where each competing team has four members that each run 400 meters, or one lap, around a standard track.  One of the two teams is very well-rounded and each of their members will run their 400 meter leg in precisely 55 seconds.  The other team is less well-rounded; their first runner will run their 400 meter leg in 60 seconds then each subsequent runner will be 3 seconds faster than the previous runner.  Using this information, how many seconds will the faster team win by?",
    # "A farmer is buying feed for his horses. He buys a variety of hay, oats, carrots and sugar cubes. Since sugar cubes are a rare treat, he only buys two 1-pound boxes of them for the whole stable. He only wants enough carrots to feed the horses while the vegetables are fresh, so he buys four 12-pound bags. Hay is the main diet of his horses, so he buys forty-two 75-pound bales. Oats are a staple to supplement the hay, so he buys twenty 65-pound sacks. If his farm truck can carry 2250 pounds at a time, how many trips does the farmer need to transport all the feed?",
    # "BEASTS OF THE SOUTHERN WILD Written by Lucy Alibar & Benh Zeitlin\n\n\nFINAL DRAFT: Based on the stage play \"Juicy and Delicious\" by Lucy Alibar \n\n\nEXT. HUSHPUPPY'S HOUSE - DAWN An abandoned looking trailer sits on top of two 15-foot-tall oil drums. Distant thunder trembles through the peeling metal panels. The structure is in such disrepair, that surely no one lives here. But then, a light goes on. \n\n\nINT. HUSHPUPPY'S HOUSE - MORNING A tiny hand sculpts the mud on top of a crawfish hole placed on the floor. We pan up to reveal a little girl examining a baby chicken that appears to be dead. This is HUSHPUPPY, an unkempt and seemingly uncared for six-year-old with a gaze of unmistakable wisdom. Hushpuppy places the chick on the crawfish hole, like a queen on her throne and the chick twitches to life, cheeps twice. Hushpuppy's esoteric science experiment is interrupted by DISTANT THUNDER. Her eyes stand to attention. \n\n\nCUT TO: EXT. SHACKO IN THE BACKO - DAY An eerie, harsh wind whips hay and dust through over a giant slumbering pot-belly pig. Hushpuppy, donned in boys' underpants and a child-sized wife- beater, tip-toes behind the epic creature. She studies it, wonderful, is this the source of the thunder? With the utmost delicacy, she lays a hand on the pig's belly. We hear his HUGE BEATING HEART. \n\n\nCUT TO: EXT. HUSHPUPPY'S HOUSE - VARIOUS A series of glimpses of Hushpuppy's scientific method. She chases geese, chickens, ducks, dogs around the property- a cross between an abandoned farm and salvage yard. Hushpuppy grabs a baby chick and puts it to her ear. A TINY HEARTBEAT. She listens with focused wonder and intensity HUSHPUPPY (V.O.) All the time, everywhere, everything's organs be beatin' and squirtin' and talkin' to each other in ways I can't understand. Mosta the time they probably just sayin' \"I'm hungry,\" or \"I gotta poop,\" but sometimes they talkin' in codes. Hushpuppy's eyes dart up as a MAN'S VOICE yells- MAN'S VOICE (O.S.) Get up, get out of here! We hear a YOWL as a cat is flung across the room. A window made from a gas station sign opens in a Robinson Crusoe style tree house patched together from storm debris and discarded appliances. A wild man with severe features, a frazz of unkempt hair, and brawler's scars opens a window made from a metal sign. This is WINDELL EMMETT DOUCET, known to all as WINK .\n\n\nWINK: Get your pants on, man! Wink kills a beer and sends it out the hole in the wall into a basketball hoop attached to a fishing net that stretched 15 feet down to the ground. The net is overflowing with beer cans. \n\n\n\nEXT. HUSHPUPPY'S HOUSE - DAY Hushpuppy obediently climbs a series of increasingly bigger and bigger oil drums that function like a ladder, up to the door of her house. \n\n\nEXT. SHACKO IN THE BACKO - DAY We now see Wink in his morning ritual. He opens a cooler with a butchered chicken inside it and tosses the bird on the grill. He goes to the front porch and pours down a bag of dog food. He pulls a clothesline that leads to Hushpuppy's trailer. A BELL RINGS\n\n\nWINK: Feed up time! Feed up! Hushpuppy, now wearing pants and a slightly more proper T- shirt, comes running down the oil barrels to her house. She echoes her Daddy.\n\n\nHUSHPUPPY: Feed up time! Feed up! \n\n\nINT. ABANDONED BUS - DAY Hushpuppy devours the whole roast chicken with her hands, getting right in there like it was a candy apple. It looks really yummy. Pigs, dogs, chickens, and cats are chowing down all around her. A hatch opens above her and we realize we're in the bottom of the Shacko. Wink sticks his head through the hatch and throws corn to the chickens.\n\n\nWINK: Share with the dogs. Hushpuppy rips off a piece of chicken and flings it to a filthy chihuahua with no hair on the back half of its body. This is WINDELL, in spite of her circumstances, quite a handsome young pup. \n\n\nEXT. BATHTUB MARSH WATER - THE TURCK - EVENING WIDE, we see Hushpuppy and Wink drift through the marsh. They ride in a severed truck bed floating on top of oil drums. A motor is strapped to the back. A sign on the boat reads \"The Turck\", Hushpuppy's spelling of \"Truck\". They look out to where the water goes all the way to a monolithic, 20 foot wall stretching infinitely into the distance. This levee, the first piece of modern construction we've seen, encloses the civilized world, protecting it from rising water. The Bathtub is on the wrong side of wall. Wink stares out at the distant factories behind the wall with a peaceful and confident disdain. Hushuppy matches his relaxed defiant expression\n\n\nWINK: Ain't that ugly over there? He takes a long pull on his beer.\n\n\nWINK: (CONT'D) We live in the prettiest place on earth. Hushpuppy looks over the wall to the Dry Side. It's an endless sprawl of oil processing power plants without a tree or bird in sight. This is the engine that runs the Northern world. \n\n\nHUSHPUPPY (V.O.) Daddy says, up above the levee, on the dry side, they're afraid of the water like a bunch of babies. They built the wall that cuts us off. \n\n\nEXT. AERIAL BATHTUB - DAY We fly over marsh and water, coming upon a tiny crop of shanties on an island perched at the very bottom of the land. \n\n\nHUSHPUPPY (V.O.) They think we all gonna drown down here. But we ain't goin' nowhere. \n\n\nCUT TO: EXT. CAUSEWAY - PRE-FLOOD - DAY CLOSE ON Hushpuppy letting out a WARRIOR SCREAM in a mass of HOLLERING people. They stand",
]

# with open('test_32k.jsonl') as f:
#     data = json.load(f)
# prompts = [
#     data['question']
# ]

NUM_REQ = 1        
input_len = 32 * 1024
# prompts = generate_prompts_auto(input_len, NUM_REQ)


# USE_LONG_PROMPT = True                                                 # True: 从 LONG_PROMPT_FILE 加载;False: 用内置短 prompt
# LONG_PROMPT_FILE = "/home/z00845282/kvoffload/script/prompts/batch-30k/longprompt1 agent.txt"  # 单一数据源
# # ============================ 构造 prompts ============================
# if USE_LONG_PROMPT:
#     if not os.path.isfile(LONG_PROMPT_FILE):
#         raise FileNotFoundError(f"prompt file not found: {LONG_PROMPT_FILE}")
#     with open(LONG_PROMPT_FILE, "r", encoding="utf-8") as f:
#         content = f.read()
#     prompts = [content] * NUM_REQ
#     print(f">>>>> loaded prompt: {LONG_PROMPT_FILE} (chars={len(content)}) x{NUM_REQ}")
# else:
#     prompts = ["The capital of France is"] * NUM_REQ



# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=50, temperature=0)
# sampling_params = SamplingParams(max_tokens=20, temperature=0.8, top_p=0.95)
# sampling_params = SamplingParams(max_tokens=200, temperature = 0.6, top_k = 40, top_p = 0.95, repetition_penalty = 1.03, ignore_eos=True)

llm = LLM(
    model=model_path,
    # trust_remote_code=True,
    # enforce_eager=True,
    # compilation_config={"cudagraph_mode":"PIECEWISE"},
    compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1, 2, 4]},
    tensor_parallel_size=16,
    # decode_context_parallel_size=2,
    # prefill_context_parallel_size=2,
    # cp_kv_cache_interleave_size=128,
    enable_expert_parallel=True,
    # enable_chunked_prefill=False,
    # enable_prefix_caching=False,
    # gpu_memory_utilization=0.9,
    # gpu_memory_utilization=0.76, # tp1, 6 layers, 32k, uti 0.76, bs 3
    gpu_memory_utilization=0.862, # tp2, 10 layers, 32k, uti 0.86, bs 1
    # gpu_memory_utilization=0.84, # tp2, 10 layers, 32k, graph, uti 0.84, bs 1
    # gpu_memory_utilization=0.94, # tp16, full, 128k, mock prefill, uti 0.95, bs1
    quantization="ascend",
    max_num_seqs=4,
    # max_model_len=4096,
    # max_num_batched_tokens=4096,
    max_model_len=33792,
    max_num_batched_tokens=33792,
    # max_model_len=82000,
    # max_num_batched_tokens=32768,
    additional_config={
        # "torchair_graph_config":{
        #     "enabled": True,
        # },
        # "ascend_scheduler_config": {
        #     "enabled": True,
        #     'chunked_prefill_enabled': False,
        # },
        # 'refresh': True,
        "use_offload": True,
        "lru_resident_cache_config": {
            "enabled": True,
            "buffer_size": 4096,
            "topk": 2048,
        },
        # "expert_offload_config": {
        #     "expert_offload": True,
        #     "num_device_experts": 6,
        # },
    },
    # speculative_config={
    #     "num_speculative_tokens": 1,
    #     "method": "deepseek_mtp"
    # },
    # kv_transfer_config = {
    #     "kv_connector": "AscendStoreConnector",
    #     "kv_role": "kv_both",
    #     "kv_connector_extra_config": {
    #     "backend": "memcache",
    #     "mooncake_rpc_port":"0",
    #     "use_layerwise": True,
    #     "discard_partial_chunks": False,
    #     }
    # },
    kv_transfer_config = {
        "kv_connector": "SFAKVOffloadConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "use_layerwise": True,
        }
    },
    block_size=128,
    async_scheduling=False,
    disable_hybrid_kv_cache_manager=False,
    profiler_config={
        "profiler": "torch",
        # "torch_profiler_dir": "/home/z00911889/profile/v32_l10_tp2_baseline_bs4_seq256_graph",
        "torch_profiler_dir": "/home/s886374/profile/glm_lru_tp16_bs1_graph",
        "torch_profiler_with_stack": True,
    }
)

for i in range(1):
    outputs = llm.generate(prompts, sampling_params)

# Generate texts from the prompts.
llm.start_profile()
outputs = llm.generate(prompts, sampling_params)
llm.stop_profile()

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
