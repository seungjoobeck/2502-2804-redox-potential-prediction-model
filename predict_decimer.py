import sys
import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from DECIMER import predict_SMILES
    print("DECIMER 로딩 성공!")
except ImportError as e:
    print(f"DECIMER 로딩 실패. (에러: {e})")
    sys.exit()

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

#여러 개의 3D 구조 분신(컨포머)들을 생성하는 함수로 변경
def get_ensemble_coords(smiles, num_confs=10, max_atoms=50):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: 
        return [np.zeros((max_atoms, 3)) for _ in range(num_confs)]
    mol = Chem.AddHs(mol)
    
    # num_confs 수만큼 서로 다른 3D 기하학 구조를 무작위 생성
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=AllChem.ETKDGv3())
    if len(conf_ids) > 0:
        try: 
            AllChem.MMFFOptimizeMoleculeConfs(mol) # 구조 최적화
        except: 
            pass
    
    all_coords = []
    confs = list(mol.GetConformers())
    
    for conf in confs:
        coords = [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] 
                  for i in range(min(mol.GetNumAtoms(), max_atoms))]
        coords = np.array(coords)
        if len(coords) < max_atoms:
            coords = np.vstack([coords, np.zeros((max_atoms - len(coords), 3))])
        all_coords.append(coords)
        
    # 혹시 모자라면 패딩 처리
    while len(all_coords) < num_confs:
        all_coords.append(np.zeros((max_atoms, 3)))
        
    return all_coords[:num_confs]

def predict_with_ensemble(image_path, num_confs=10):
    print(f"이미지 구조 분석 중: {image_path}")
    try:
        smiles = predict_SMILES(image_path)
        print(f"인식된 SMILES: {smiles}")
    except Exception as e:
        print(f"이미지 인식 실패: {e}")
        return

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print("유효하지 않은 SMILES입니다.")
        return

    # 공통 특징 추출 (2D 지문 및 추가 정보는 동일)
    fp = torch.FloatTensor(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))).unsqueeze(0)
    bonds = [b.GetBondType() for b in mol.GetBonds()]
    extra = torch.FloatTensor([[
        bonds.count(Chem.rdchem.BondType.SINGLE),
        bonds.count(Chem.rdchem.BondType.DOUBLE),
        bonds.count(Chem.rdchem.BondType.TRIPLE),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ]])

    # 모델 로드
    try:
        model = AutonomousHybridNet(extra_feat_dim=4)
        model.load_state_dict(torch.load('orp_robust_v1.pth', map_location='cpu'))
        model.eval()
        scaler = joblib.load('orp_robust_scaler.pkl')
        
        # 10개의 서로 다른 3D 기하학 구조를 가져옴
        print(f"{num_confs}개의 독립된 3D 기하학 구조 생성 및 다중 교차 계산 중...")
        coords_list = get_ensemble_coords(smiles, num_confs=num_confs)
        
        predictions = []
        with torch.no_grad():
            for c in coords_list:
                coords_tensor = torch.FloatTensor(c).unsqueeze(0)
                pred_scaled = model(fp, coords_tensor, extra).numpy()
                pred_mv = scaler.inverse_transform(pred_scaled)[0][0]
                predictions.append(pred_mv)
        
        #최종 결과 통계 및 평균 계산
        final_mean = np.mean(predictions)
        final_std = np.std(predictions)
        
        print(f"분자 SMILES: {smiles}")
        print(f"{num_confs}번 계산된 원시값 리스트: {[round(p, 1) for p in predictions]}")
        print(f"최종 평균 예측값: {final_mean:.2f} mV (신뢰 오차 범위: ±{final_std:.2f} mV)")
        print("="*45)
        
    except Exception as e:
        print(f"전위값 계산 중 오류 발생: {e}")

if __name__ == "__main__":
    target_image = "test_molecule4.jpg" # 분석 타겟 이미지 명
    
    if os.path.exists(target_image):
        # num_confs=10 으로 10번 교차 검증해서 평균 때림 (필요시 20이나 30으로 늘려도 됨)
        predict_with_ensemble(target_image, num_confs=10)
    else:
        print(f"'{target_image}' 파일이 없습니다.")
