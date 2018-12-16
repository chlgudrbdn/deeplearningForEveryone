# https://m.blog.naver.com/PostView.nhn?blogId=joooople&logNo=221240814631&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F
import numpy as np


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율
    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 매수 또는 매도 수수료 0.015%
    TRADING_TAX = 0.003  # 거래세 0.3%
    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    ACTIONS = [ACTION_BUY, ACTION_SELL]  # 인공 신경망에서 확률을 구할 행동들
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self, index, environment,
                 min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=.05):
        self.index = index
        # self.state = None
        # self.signal = 0
        # self.signalLog = []
        # self.input_data = None
        # self.output_data = None

        # Environment 객체
        self.environment = environment  # 현재 주식 가격을 가져오기 위해 환경 참조

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        self.delayed_reward_threshold = delayed_reward_threshold  # 지연보상 임계치

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = 0  # balance + num_stocks * {현재 주식 가격}
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율
    """
    def action_to(self, Agent, input):
        Agent.state = input

    def select_choice(self, price, n, N):  # 확률식은 초기조건 시트에 명시된 i일차 개인별 구매확률이 아닌 ppt를 따름. 단 감마라고 적힌 부분은 시트에 따라 delta로 대체
        buy_prob = (1 - (price / 2000)) * (alpha + beta * (n / N) + delta * ((math.exp(-math.exp(-0.15 * (sum(self.buyLog) - 5)))) - 0.12))
        # 구매 회수 m은 구매기록의 값을 모두 더한 값으로 한다.
        self.buyProbabilityLog.append(buy_prob)
        if random.random() < buy_prob:
            return 1
        else:
            return 0
    """
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / \
                          int(self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = self.portfolio_value / self.initial_balance
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    def decide_action(self, policy_network, sample, epsilon):
        confidence = 0.
        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)  # 무작위로 행동 결정
        else:
            exploration = False
            probs = policy_network.predict(sample)  # 각 행동에 대한 확률
            action = np.argmax(probs) if np.max(probs) > 0.1 else Agent.ACTION_HOLD  # 개미가 기관과 외국인 보다 다른 지점.
            confidence = probs[action] if action != Agent.ACTION_HOLD else 0.5  # holding일 경우가 문제라서, holding일 경우는 그냥 confidence 적당히 0.5정도로 넣음.
        return action, confidence, exploration

    def validate_action(self, action):
        validity = True
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (
                1 + self.TRADING_CHARGE) * self.min_trading_unit:
                validity = False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                validity = False
        return validity

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_traiding = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_traiding

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(min(
                    int(self.balance / (
                        curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount  # 보유 현금을 갱신
            self.num_stocks += trading_unit  # 보유 주식 수를 갱신
            self.num_buy += 1  # 매수 횟수 증가

            self.immediate_reward = 1  # 에이전트마다 조금씩 다르게.

        # 매도
        elif action == Agent.ACTION_SELL:  # sell
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
            self.balance += invest_amount  # 보유 현금을 갱신
            self.num_sell += 1  # 매도 횟수 증가

            self.immediate_reward = 1  # 에이전트마다 조금씩 다르게.

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가
            self.immediate_reward = -1

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        profitloss = (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value

        # 지연 보상 판단
        if profitloss > self.delayed_reward_threshold:
            delayed_reward = 1
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:
            delayed_reward = -1  # 에이전트마다 조금씩 다르게.
            # 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward = 0  # 에이전트마다 조금씩 다르게.
        return self.immediate_reward, delayed_reward



class AntAgent(Agent):
    def __init__(self, index):
        self.index = index


# 애널리스트(예언자)
class AnalystAgent(Agent):
    def __init__(self, index):
        self.index = index

    def lie(self, Agent):
        Agent.socket = self.signal


# 기업. 대상이 된다.
class FirmAgent(Agent):
    def __init__(self, index):
        self.index = index


# 기관투자자(증권, 등등)
class Institution(Agent):
    def __init__(self, index):
        self.index = index


"""
# 검은머리 외국인
class BlackHeadForeigner(Agent):
    def __init__(self, index):
        self.index = index


# 불개미는 ‘강세의’라는 뜻의 ‘bull’ 과 ‘개미’의 결합어로 주,식시장에서 강세장을 예상해 공격적 매수를 하는 개인투자자를 의미합니다. 시장을 낙관적(Bull)으로 보고 무섭게 뛰어든다는 뜻도 담겨있죠.
class BullAntAgent(AntAgent):
    def __init__(self, index):
        self.index = index


# 슈퍼개미는 자산 규모가 큰 개인투자자를 일컫습니다. 적게는 수십억 원에서 많게는 수백억 원을 거래하고 단순 투자를 넘어 경영참여도 시도하죠.
class SuperAntAgent(AntAgent):
    def __init__(self, index):
        self.index = index


# 왕개미는 주,식 투자로 인해 큰 돈을 벌어들인 개인투자자를 의미하는 은어
class KingAntAgent(AntAgent):
    def __init__(self, index):
        self.index = index
"""
