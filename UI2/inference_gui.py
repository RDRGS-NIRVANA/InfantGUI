import torch
import numpy as np
from network import C3D_model, Resnet_3D
import cv2
from PyQt5.QtCore import QThread, pyqtSignal
import librosa

torch.backends.cudnn.benchmark = True


def moderl_load():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    model = Resnet_3D.generate_model(50)
    checkpoint = torch.load(r'C:\Users\dev1se\Desktop\InfantGUI\run\3DResnet-ucf101_epoch-1999.pth.tar',map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    return model

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def infantDetection(model, clip, frame):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prob_result = []

    if len(clip) == 16:
        print(1)
        inputs = np.array(clip).astype(np.float32)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
        inputs = torch.from_numpy(inputs)
        inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
        with torch.no_grad():
            outputs = model.forward(inputs)
        probs = torch.sigmoid(outputs)
        prob_result = probs.cpu().detach().numpy()
        class_result = np.int64(prob_result > 0.7)

        prob_result = prob_result.reshape(prob_result.shape[0] * prob_result.shape[1], )
        print(2)
        clip.pop(0)

    return frame, class_result, prob_result

#导入音频视频
def audio_model_load():
    # 加载模型
    model_path = r'C:\Users\dev1se\Desktop\InfantGUI\audio\models\resnet18-0.96.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path)
    model.to(device)
    model.eval()
    return model

def audio_model_detection(model,data,sr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec_mag = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=1024, hop_length=348)
    spec_mag = spec_mag[:, :64]
    mean = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    spec_mag = spec_mag[np.newaxis, np.newaxis, :]

    data=torch.tensor(spec_mag,dtype=torch.float32,device=device)
    output=model(data)
    result=torch.nn.functional.softmax(output)
    result=result.data.cpu().numpy()
    return result[0]

if __name__ == '__main__':
    videoName = ""
    infantDetection(videoName)
