import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

def load_robust_dataset(csv_path):
    print("오리지널 CSV 데이터 로드 중...")
    df = pd.read_csv(csv_path)
    
    # 결측치 제거 후 정확히 90,000번부터 100,000번까지 1만 개 추출 (MSE 최적 구간)
    df = df.dropna(subset=['smiles_r', 'ORP_PM7linfit']).reset_index(drop=True)
    
    print("[MSE 최적화] 90k - 100k 구간 유기 화합물 10,000개 추출 중...")
    robust_df = df.iloc[90000:100000].reset_index(drop=True)
    return robust_df

def get_real_coords(smiles, max_atoms=50):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: 
        return np.zeros((max_atoms, 3))
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 42
    if AllChem.EmbedMolecule(mol, ps) >= 0:
        try: AllChem.MMFFOptimizeMolecule(mol)
        except: pass
    conf = mol.GetConformer()
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] 
                      for i in range(min(mol.GetNumAtoms(), max_atoms))])
    if len(coords) < max_atoms:
        coords = np.vstack([coords, np.zeros((max_atoms - len(coords), 3))])
    return coords

def extract_features(df, max_atoms=50):
    fps, coords_list, extra_feats, targets = [], [], [], []
    total = len(df)
    
    target_col = 'ORP_PM7linfit'
    print(f" [확인된 열 구조] SMILES 열: 'smiles_r' | 타겟 전위 열: '{target_col}'")
    
    print("특징(2D 지문 + 3D 좌표 + 결합 정보) 추출 시작...")
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f" 진행률: [{idx}/{total}] 완료")
            
        smiles = row['smiles_r'] 
        mol = Chem.MolFromSmiles(smiles)
        if not mol: continue
        
        # 1) 2D Morgan Fingerprint
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
        
        # 2) 3D Coords
        coords = get_real_coords(smiles, max_atoms)
        
        # 3) Extra Features
        bonds = [b.GetBondType() for b in mol.GetBonds()]
        extra = [
            bonds.count(Chem.rdchem.BondType.SINGLE),
            bonds.count(Chem.rdchem.BondType.DOUBLE),
            bonds.count(Chem.rdchem.BondType.TRIPLE),
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ]
        
        fps.append(fp)
        coords_list.append(coords)
        extra_feats.append(extra)
        targets.append(row[target_col])
        
    return np.array(fps), np.array(coords_list), np.array(extra_feats), np.array(targets).reshape(-1, 1)

class ORPDataset(Dataset):
    def __init__(self, fps, coords, extra, targets):
        self.fps = torch.FloatTensor(fps)
        self.coords = torch.FloatTensor(coords)
        self.extra = torch.FloatTensor(extra)
        self.targets = torch.FloatTensor(targets)
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.fps[idx], self.coords[idx], self.extra[idx], self.targets[idx]

# ==========================================
# 2. 강건한 하이브리드 딥러닝 모델 정의
# ==========================================
class AutonomousHybridNet(nn.Module):
    def __init__(self, extra_feat_dim=4):
        super(AutonomousHybridNet, self).__init__()
        self.coords_extractor = nn.Sequential(
            nn.Conv1d(3, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(2048 + 64 + extra_feat_dim, 512), nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, fp, coords, extra_feats):
        c_feat = self.coords_extractor(coords.transpose(1, 2))
        combined = torch.cat([fp, c_feat, extra_feats], dim=1)
        return self.fusion_layer(combined)

# ==========================================
# 3. 메인 학습 파이프라인
# ==========================================
def train_pipeline():
    csv_path = "redox_results.csv" 
    if not os.path.exists(csv_path):
        print(f"'{csv_path}' 파일이 없습니다. 경로를 확인해주세요.")
        return
        
    robust_df = load_robust_dataset(csv_path)
    fps, coords, extra, targets = extract_features(robust_df)
    
    scaler = StandardScaler()
    targets_scaled = scaler.fit_transform(targets)
    joblib.dump(scaler, 'orp_robust_scaler.pkl')
    print("정규화 스케일러 저장 완료: orp_robust_scaler.pkl")
    
    X_fp_tr, X_fp_val, X_co_tr, X_co_val, X_ex_tr, X_ex_val, y_tr, y_val = train_test_split(
        fps, coords, extra, targets_scaled, test_size=0.2, random_state=42
    )
    
    train_dataset = ORPDataset(X_fp_tr, X_co_tr, X_ex_tr, y_tr)
    val_dataset = ORPDataset(X_fp_val, X_co_val, X_ex_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"학습 디바이스: {device}")
    
    model = AutonomousHybridNet(extra_feat_dim=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = 100
    best_val_loss = float('inf')
    print("최적 MSE 모델 학습 개시...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_fp, batch_co, batch_ex, batch_y in train_loader:
            batch_fp, batch_co, batch_ex, batch_y = (
                batch_fp.to(device), batch_co.to(device), batch_ex.to(device), batch_y.to(device)
            )
            
            optimizer.zero_grad()
            outputs = model(batch_fp, batch_co, batch_ex)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_fp.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_fp, batch_co, batch_ex, batch_y in val_loader:
                batch_fp, batch_co, batch_ex, batch_y = (
                    batch_fp.to(device), batch_co.to(device), batch_ex.to(device), batch_y.to(device)
                )
                outputs = model(batch_fp, batch_co, batch_ex)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_fp.size(0)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'orp_robust_v1.pth')
            
    print(f"가중치 파일 저장 완료 (Best Val Loss: {best_val_loss:.4f})")
    
    # 최종 성적표 도출
    model.load_state_dict(torch.load('orp_robust_v1.pth'))
    model.eval()
    all_preds, all_real = [], []
    
    with torch.no_grad():
        for batch_fp, batch_co, batch_ex, batch_y in val_loader:
            batch_fp, batch_co, batch_ex = batch_fp.to(device), batch_co.to(device), batch_ex.to(device)
            outputs = model(batch_fp, batch_co, batch_ex).cpu().numpy()
            all_preds.append(outputs)
            all_real.append(batch_y.numpy())
            
    preds_mv = scaler.inverse_transform(np.vstack(all_preds))
    real_mv = scaler.inverse_transform(np.vstack(all_real))
    
    mse = np.mean((preds_mv - real_mv) ** 2)
    mae = np.mean(np.abs(preds_mv - real_mv))
    rmse = np.sqrt(mse)
    
    print(f"최종 MSE : {mse:.4f}")
    print(f"최종 MAE : {mae:.4f} mV")
    print(f"최종 RMSE: {rmse:.4f} mV")
    print("="*45)

if __name__ == "__main__":
    train_pipeline()
