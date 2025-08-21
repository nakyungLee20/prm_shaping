#!/bin/sh
#SBATCH -J jupyter
#SBATCH -p gpu03
#SBATCH --gres=gpu:1
#SBATCH --ntasks=12
#SBATCH -t 12:00:00 
#SBATCH -o /home/leena/prm_shaping/logs/%x_%A.out

## 1. 현재 작업이 실행 중인 노드의 호스트 이름(주소)을 가져옵니다.
NODE_IP=$(hostname -I | awk '{print $1}')

## 2. 연결에 필요한 정보를 터미널과 로그 파일에 명확하게 출력합니다.
echo "===================================================================="
echo "VS Code Jupyter 서버가 아래 주소에서 실행 중입니다."
echo "Jupyter 서버 URL (이 주소를 복사해서 VS Code에 붙여넣으세요):"
echo "http://${NODE_IP}:8080"
echo "===================================================================="

## 3. Jupyter Notebook 서버를 실행합니다.
# --ip=0.0.0.0 : 모든 IP 주소에서의 연결을 허용합니다.
# --port=8080 : 8080 포트를 사용합니다.
# --no-browser : 서버에서 웹 브라우저를 자동으로 열지 않습니다.
# --allow-root : (필요 시) root 계정으로 실행을 허용합니다.
jupyter notebook --ip=0.0.0.0 --port=8080 --no-browser