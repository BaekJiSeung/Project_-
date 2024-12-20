이 코드는 감염병(measles)의 혈청 유병률(seroprevalence) 데이터를 분석하여 감염력(force of infection, FOI)을 추정하는 것을 목표로 한다.  FOI는 질병이 특정 인구 내에서 전파되는 속도를 나타내는 파라미터이다. 이를 통해 전염병의 전파 특성을 이해하고, 감염 확산을 예측하고 그를 예방하고 대처 방안을 수립하는 것에 도움이 된다.

Motivation

force of infection(FOI)sms 전염병의 전파를 이해 예측하는것에 핵심적 옇할을 한다. 특이 특정 연령대에서 감염이 발생하는 비율을 파악하는데 사용되기도 한다. 혈청 유병률(seroprevalence) 는 말 그대로 특정 연령대의 사람들을 검사하여 그중 얼마만큼의 비율의 사람들이 항체를 보유하고 있는지 즉 과거의 감염 여부를 파악함으로써  감염 나이, 감염 정도, 주기성 등의 정보를 알수 있다. 이 코드에서는 데이터와 머신러닝 기법 이용하여 FOI를 추정하고 전염률을 계산하여 질병으ㅔ 대한 예측을 하는 것이다.

Data

데이터는 간단하게 연령에 따른 유병률을 영국과 중국에서 각각 조사한 csv파일 2개를 사용하였다.
각 데이터는 특정연령대, 연령대별 양성으로 판정된 사람 수, 연령대 별 조사한 사람 수로 이루어져 있고 이를 통해 연령대별 과거 감염 비율을 코드 내에서 계산하여 이용하였다.

배경지식
SEIR 모델은 전염병의 유행을 시뮬레이션하기 위해 사용되는 수학적 모델링으로 네 가지 상태로 구성된다.
S: 감염 가능성이 있는 사람들 (Susceptible) E: 잠복기 상태의 사람들 (Exposed) I: 감염된 사람들 (Infectious) R: 회복된 사람들 (Recovered) dSdt = -beta * S * I dEdt = beta * S * I - sigma * E dIdt = sigma * E - gamma * I dRdt = gamma * I

여기서 β는 전염율, σ는 잠복기 상태에서 감염 상태로의 전환율, γ는 회복율을 나타낸다. measles의 잠복기는 8일이고 평균 기간은 7일이므로 σ,γ는 각각 역수인 1/8, 1/7 이다. 유병률을 통해 구한 람다의 추정값으로 베타를 구할 수가 있다. 이 베타는 질병을 연구하는데에 중요한 숫자인 R0로 이어진다.

기본 재생산지수 R0: 이 값은 전염병의 전파력을 나타내며, 한 명의 감염자가 몇 명의 사람을 감염시킬 수 있는지를 의미한다. 만약 이 값이 1보다 크면 전염병이 확산되고, 1보다 작으면 전염병이 확산하지 않는다. 

집단 면역 (Herd Immunity): 특정 인구가 집단 면역을 달성하기 위해 얼마나 많은 사람이 면역이 되어야 하는지 결정하는 데 도움을 준다. H = 1 - 1/R0 의 값을 가진다. 예를 들어 H = 0.7이라면 전체의 70퍼센트만 접종을 통해 면역력을 갖추면 더 이상의 전염이 일어나지 않는다는 뜻이다. 이는 예방 접종 보건 정책 수립에 유용한 정보를 제공한다.

베이지안 추정
데이터와 사전 지식(혹은 사전 분포)을 베이즈 정리를 통해 결합하여 파라미터에 대한 추정을 수행하는 방법이다. 한마디로 파라미터의 값이 하나로 정해져있고 그것을 추정하는 것이 아니라 파라미ㅓ 값이 어떠한 분포를 지닌다는 가정 하에 그 분포에 대해서 추정을 하는 것이다. 베이지안 추정에서는 사전 분포(prior distribution), likelihood를 결합하여 Posterior Distribution을 얻을 수 있다. 


metropolis hastings
메트로폴리스-헤이스팅스 알고리즘은 확률 분포에서 샘플을 생성하는 마르코프 체인 몬테카를로 (MCMC) 방법 중 하나이다. 이 알고리즘은 주어진 확률 분포에 대해 독립된 샘플을 생성하기 어려울 때 사용되며, 특히 베이지안 추정에서 널리 사용된다. 초기 파라미터 값을 선택하고 그 값에서 확률적으로 나올 수 있는 새로운 값을 찾아낸다. 그 값이 기존에 주어졌던 값과 비교하여 likelihood가 나은지 판단하여 iteration을 통해 샘플을 생성해낸다. 이 코드에서는  메트로폴리스-헤이스팅스 모듈을 만들었고 그 안에서 카탈리틱 likelihood를 통하여 파라미터의 likelihood를 계산하여 알고리즘을 적용시켰다.

HPD & ESS
HPD는 베이지안 추론에서의 credible interval읭 역할을 한다. posterior 분포에서 밀도가 높은 구간을 나타낸다. 보통 95퍼센트의 확률을 만족하는 구간으로 사용한다. 코드에서는 MCMC 샘플에서의 HPD 신뢰구간을 계산하였고 이를 통해 FOI의 정확도를 수치화할수 있었다.
ESS는 MCMC샘플이 얼마나 독립적인지를 나타낼수 있는 지표이더. MCMC에서는 파라미터의 값이 다음 파라미터를 결정하기 때문에 완전히 독립된 샘플들보다 정보량이 적을수 박에 없다. ESS값이 낮으면 샘플간의 상관관계가 크게 되어 샘플의 필요량이 더 늘어나게 된다.

코드설명
main.ipynb 와 모듈 metropolis_hastings.py, utils.py로 이루어져 있다.
영국,중국의 혈청유병률데이터를 불러온다. 각 데이터에서 연령대, 감염수, 총 표본 수를 추출한다. 
데이터를 통해 메트로폴리스헤이스팅스 클래스를 사용하여 각 데이터셋에서 파라미터인 lamda의 분포를 추정한다. 알고리즘을 사용한 샘플에서 burn in을 제거하고, 샘플이 얼마나 효과적인지 utils.py 모듈에서의 ESS 를 통하여 샘플이 효과적인 정도를 계산한다. 또한 파라미터에 대한 psterior분포를 히스토그램으로 시각화하고 확률분포의 신뢰구간을 계산한다.




