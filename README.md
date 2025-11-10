# Project AIpex

* AI Object Detection이 탑재된 스마트 헬맷 올빼미 개발

## Concept arts

<img src="https://github.com/AIpex-sesac/AIpex/blob/main/sources/concept_art.png?raw=true" width="300px"/>
![concept demo](https://github.com/AIpex-sesac/AIpex/blob/main/sources/concept_video.gif?raw=true)

## High Level Design

* 헬멧의 스크린을 통해 전방 Object Detection과 그 결과를 활용한 증강현실을 착용자에게 제공
* 네비게이션, 후방 카메라 등 자동차에 탑재되어 있는 편의기능을 헬멧에 탑재하여 착용자에게 제공
## Clone code

* (각 팀에서 프로젝트를 위해 생성한 repository에 대한 code clone 방법에 대해서 기술)

```shell
git clone https://github.com/AIpex-sesac/AIpex.git
```

## Prerequite

* (프로잭트를 실행하기 위해 필요한 dependencies 및 configuration들이 있다면, 설치 및 설정 방법에 대해 기술)

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Steps to build

* (프로젝트를 실행을 위해 빌드 절차 기술)

```shell
cd ~/xxxx
source .venv/bin/activate

make
make install
```

## Steps to run

* (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

```shell
cd ~/xxxx
source .venv/bin/activate

cd /path/to/repo/xxx/
python demo.py -i xxx -m yyy -d zzz
```

## Output

* (프로젝트 실행 화면 캡쳐)
* 현재 샘플 디자인

![./sample.jpg](./sources/design_sample.png)

## Appendix
```mermaid
flowchart LR
    A[Camera] -->|영상 촬영| B[pi]
    B -->|영상 제공| C[AI model]
    C -->|Object Detection| D[result]
    D -->|결과 출력| E[(Display)]
```
* (참고 자료 및 알아두어야할 사항들 기술)

## Dataset and Model
* 팀원들이 직접 카메라가 부착된 헬멧을 착용하고 한강변을 주행하여 데이터 수집
* intel geti를 활용하여 labeling 진행
### Model
* ATSS-MobileNetV2
### Classes
1. bike
2. person
3. car

## 팀원 소개 및 역할 분담
  | Name | Role |
  |----|----|
  | 남대문 | Development |
  | 성시경 | Data management, 3D Modeling |
  | 장태규 | Project lead, Architect |
  | 최종인 | UI design, Development |