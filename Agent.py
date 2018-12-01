# https://m.blog.naver.com/PostView.nhn?blogId=joooople&logNo=221240814631&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F


class Agent:
    def __init__(self, index):
        self.index = index
        self.state = None
        self.signal = 0
        self.signalLog = []
        self.input_data = None
        self.output_data = None

    def action_to(self, Agent, input):
        Agent.state = input

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
