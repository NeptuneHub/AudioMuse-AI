import sys, os
# ensure parent workspace directory is in path so student_clap package loads
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch, yaml, numpy as np
from student_clap.models.student_onnx_model import StudentCLAPAudio

cfg = yaml.safe_load(open('config.yaml'))
print('cfg keys', cfg.keys())
print('cfg content snippet', {k: cfg[k] for k in list(cfg.keys())[:5]})
# StudentCLAPAudio expects the full config dict (audio, model, etc.)
model = StudentCLAPAudio(cfg)
ckpt_path = '../model/epoch_36.pth'
print('loading checkpoint', ckpt_path)
ckpt_data = torch.load(ckpt_path, map_location='cpu')
print('checkpoint keys', list(ckpt_data.keys()))
# common patterns: model_state_dict or state_dict
if 'model_state_dict' in ckpt_data:
    sd = ckpt_data['model_state_dict']
elif 'state_dict' in ckpt_data:
    sd = ckpt_data['state_dict']
else:
    sd = ckpt_data
# remove possible prefixes
newsd = {k.replace('model.', ''): v for k, v in sd.items()}
print('adjusted keys sample', list(newsd.keys())[:10])
model.load_state_dict(newsd, strict=False)  # allow mismatched speciality/student prefixes
model.eval()

out_path = '../model/model_epoch_36_batched.onnx'
print('exporting to', out_path)
dummy = torch.randn(1,1,model.n_mels,1000)
import onnxruntime as ort

torch.onnx.export(
    model,
    dummy,
    out_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['mel_spectrogram'],
    output_names=['embedding'],
    dynamic_axes={'mel_spectrogram': {0: 'batch_size', 3: 'time_frames'},
                  'embedding': {0: 'batch_size'}}
)

sess = ort.InferenceSession(out_path)
print('session input shape', sess.get_inputs()[0].shape)

# verify
batch = np.random.randn(3,1,model.n_mels,1000).astype(np.float32)
out = sess.run(None, {'mel_spectrogram': batch})[0]
print('batched output', out.shape)
orig_sess = ort.InferenceSession('../model/model_epoch_36.onnx')
outs = [orig_sess.run(None, {'mel_spectrogram': batch[i:i+1]})[0] for i in range(batch.shape[0])]
outs = np.vstack(outs)
diff = np.abs(out - outs).max()
print('max diff', diff)
