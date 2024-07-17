import re
import random
import time
from statistics import mode
import os

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

torch.cuda.empty_cache()
import gc
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# 画像のデータ拡張を行う関数を定義

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    # 文字列の前処理を行う関数
    text = text.lower()

    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    text = re.sub(r'\b(a|an|the)\b', '', text)

    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    text = re.sub(r"[^\w\s':]", ' ', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+', ' ', text).strip()
        # ストップワードの削除
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # 単語のレンマ化 (Lemmatization)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # 単語のステミング (Stemming)
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]

    # 最終的なテキストを再構築
    text = ' '.join(stemmed_words)

    return text

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pandas.read_json(df_path)
        self.answer = answer

        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}

        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

    def update_dict(self, dataset):
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        #7.16一旦消すimage_path = f"{self.image_dir}/{self.df['image'][idx]}"
        #7.16
        image_path = f"/home/whill/desktop/pm/data/train/{self.df['image'][idx]}"
        
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            #print(f"Image not found: {image_path}. Searching from newdata...")
            image_path = f"/home/whill/desktop/pm/data/newdata/{self.df['image'][idx]}"
            try:
                image = Image.open(image_path)
            except FileNotFoundError:
            #もとのやつ　image_path = f"/home/whill/desktop/pm/data/train/{self.df['image'][idx]}"
                if idx>19872:
                    idx=idx-19873#idxが19774~と続いてしまう対策
                filename = f"train_{idx:05d}.jpg"  # idxを使ってファイル名を構成する
            # ファイルパスの生成
                image_path = f"/home/whill/desktop/pm/data/train/{filename}"
            #7.16
    
    
            image = Image.open(image_path)


        image = self.transform(image)

        question = np.zeros(len(self.idx2question) + 1)
        question_words = self.df["question"][idx].split(" ")
        for word in question_words:
            try:
                question[self.question2idx[word]] = 1
            except KeyError:
                question[-1] = 1

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)

            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)
#7.15 koko generate newdata
    def generate_new_data(self, new_data_folder):
        print("resize start")
        os.makedirs(new_data_folder, exist_ok=True)
        for idx in range(len(self.df)):
            image_path = f"{self.image_dir}/{self.df['image'][idx]}"
            image = Image.open(image_path)
            target_size = (1200, 1200)#1200にするtarget_size = (1200, 1200)
            random_scale = random.uniform(0.8, 0.9)
            image = image.resize(target_size, Image.LANCZOS)#実際にサイズ変更
            new_size = (int(image.size[0] * random_scale), int(image.size[1] * random_scale))
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=new_size)
            cropped_image = F.crop(image, i, j, h, w)

            new_image_filename = f"new_image_{idx}.jpg"
            new_image_path = os.path.join(new_data_folder, new_image_filename)
            cropped_image.save(new_image_path)
            print(f"Processed image {idx}/{len(self.df) - 1}")  # ループのインデックスを表示
             #print("resize finish")

             # データフレームに新しい画像の情報を追加する
            self.df.loc[len(self.df)] = {
                "image": new_image_filename,
                "question": self.df["question"][idx],
                "answers": self.df["answers"][idx]
            }

            #7.15
            

# 2. 評価指標の実装
# 簡単にするならBCE（簡単な損失関数）を利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        #self.resnet = ResNet18() 2024.7.12  
        self.resnet = ResNet50() 
        self.text_encoder = nn.Linear(vocab_size, 512)

        self.fc = nn.Sequential(
            #nn.Linear(1024, 512), 2024.7.12
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            #nn.Linear(512, n_answer) 2024.7.12
            nn.Linear(1024, n_answer) 
        )

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量
        question_feature = self.text_encoder(question)  # テキストの特徴量

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    gc.collect()#2024.7.12
    torch.cuda.empty_cache()#2024.7.12
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy
    gc.collect()#2024.7.12
    torch.cuda.empty_cache()#2024.7.12
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

print("OK")

            
            
            
print("ok")   
# メイン関数
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = VQADataset(df_path="/home/whill/desktop/pm/data/train.json", image_dir="/home/whill/desktop/pm/data/train", transform=transform)
    train_dataset.generate_new_data("/home/whill/desktop/pm/data/newdata")

    test_dataset = VQADataset(df_path="/home/whill/desktop/pm/data/valid.json", image_dir="/home/whill/desktop/pm/data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VQAModel(vocab_size=len(train_dataset.question2idx) + 1, n_answer=len(train_dataset.answer2idx)).to(device)

    num_epoch = 40
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(num_epoch):
        print("learning start")
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

        model.eval()
        submission = []
        for image, question in test_loader:
            image, question = image.to(device), question.to(device)
            pred = model(image, question)
            pred = pred.argmax(1).cpu().item()
            submission.append(pred)

        submission = [train_dataset.idx2answer[id] for id in submission]
        submission = np.array(submission)
        torch.save(model.state_dict(), "model.pth")
        np.save("submission7.17.last.npy", submission)

if __name__ == "__main__":
    main()

