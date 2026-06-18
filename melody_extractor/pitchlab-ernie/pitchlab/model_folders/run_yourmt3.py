from mt3_infer import load_model

# Load model explicitly (cached for reuse)
def run_mt3_pytorch(audio, params=None):
    model = load_model("mt3_pytorch", device="cuda")
    midi = model.transcribe(audio, sr=16000)

def run_mr_mt(audio, params=None):
    model = load_model("mr_mt3", device="cuda")
    midi = transcribe(audio, model="mr_mt3")

def run_yourmt3(audio, params=None):
    model = load_model("yourmt3", device="cuda")
    midi = mode.transcribe(audio, model="yourmt3")
