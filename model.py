import torch
import torch.nn as nn
import torchvision.models as models

class PacePredictionModel(nn.Module):
    def __init__(self, num_athletes):
        super(PacePredictionModel, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_athletes)
    
    def forward(self, frames, coords):
        batch_size, seq_len, _, _, _ = frames.size()
        rois = [self.extract_rois(frames[i], coords[i]) for i in range(batch_size)]
        features = [self.feature_extractor(roi) for roi in rois]
        features = torch.stack(features)
        rnn_out, _ = self.rnn(features)
        paces = self.fc(rnn_out[:, -1, :])
        return paces
    
    def extract_rois(self, frame, coords):
        rois = []
        for coord in coords:
            x1, y1, x2, y2 = coord
            roi = frame[:, y1:y2, x1:x2]
            roi = nn.functional.interpolate(roi, size=(224, 224))  # Resize to match input size of ResNet
            rois.append(roi)
        return torch.stack(rois)