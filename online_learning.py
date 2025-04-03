# -*- coding: utf-8 -*-
import os
import requests
import json
from dotenv import load_dotenv
import websocket
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import MlpPolicy
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import threading
import time
import datetime

# 추가 라이브러리: 미국 증시 캘린더 확인용
import pandas_market_calendars as mcal
import pytz

# === 환경 변수 로드 ===
load_dotenv()
ACCOUNT_NO = os.getenv("ACCOUNT_NO")
APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
ACCESS_TOKEN = None

# === KIS API 클라이언트 ===
class KISClient:
    def __init__(self):
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.token = ACCESS_TOKEN or self.get_access_token()
        self.headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.token}",
            "appkey": APP_KEY,
            "appsecret": APP_SECRET
        }
        self.ws = None

    def get_access_token(self):
        url = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": APP_KEY,
            "appsecret": APP_SECRET
        }
        response = requests.post(url, json=body)
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            print("[KIS] Access token 발급 완료")
            return self.token
        raise Exception(f"토큰 발급 실패: {response.text}")

    def get_price(self, symbol):
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/price"
        params = {"EXCD": "NAS", "SYMB": symbol}
        headers = self.headers.copy()
        headers["tr_id"] = "HHDFS00000300"  # 해외 주식 현재가 조회
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"[KIS] {symbol} 가격 조회 성공: {data}")
            last_price = data["output"].get("last", "")
            try:
                return float(last_price) if last_price else 1.0
            except (ValueError, TypeError):
                print(f"[KIS] {symbol} 가격 변환 실패: last={last_price}, 기본값 1.0 반환")
                return 1.0
        print(f"[KIS] {symbol} 가격 조회 실패: {response.status_code} - {response.text}")
        return 1.0

    def place_order(self, symbol, side, qty):
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        body = {
            "CANO": ACCOUNT_NO[:8],
            "ACNT_PRDT_CD": "01",
            "PDNO": symbol,
            "ORD_DVSN": "01",
            "ORD_QTY": str(qty),
            "OVRS_EXCG_CD": "NAS",
            "SLL_BUY_DVSN_CD": "01" if side == "buy" else "02"
        }
        response = requests.post(url, headers=self.headers, json=body)
        if response.status_code == 200:
            print(f"[KIS] {symbol} {side} 주문 성공: {qty}주")
            return True
        print(f"[KIS] {symbol} {side} 주문 실패: {response.text}")
        return False

    def on_message(self, ws, message):
        data = json.loads(message)
        print(f"[KIS WebSocket] 실시간 데이터: {data}")

    def start_websocket(self):
        ws_url = "wss://openapi.koreainvestment.com:9443/websocket"
        self.ws = websocket.WebSocketApp(
            ws_url,
            header={"Authorization": f"Bearer {self.token}"},
            on_message=self.on_message
        )
        threading.Thread(target=self.ws.run_forever).start()

# === 미국 증시가 열려 있는지 확인하는 함수 ===
def is_us_market_open():
    nyse = mcal.get_calendar('NYSE')
    today = datetime.datetime.now().date()
    # 해당 날짜에 대한 스케줄 조회 (주말, 공휴일이면 스케줄이 비어 있음)
    schedule = nyse.schedule(start_date=today, end_date=today)
    if schedule.empty:
        return False
    # 미국 동부 시간대로 변환
    market_open = schedule.iloc[0]['market_open'].tz_convert('US/Eastern')
    market_close = schedule.iloc[0]['market_close'].tz_convert('US/Eastern')
    now = datetime.datetime.now(pytz.timezone('US/Eastern'))
    return market_open <= now <= market_close

# === 온라인 거래 환경 ===
# 오프라인 학습과 동일하게 5차원 관측값 사용: [SOXL 가격, SOXS 가격, SOXL 보유량, SOXS 보유량, 현금]
class OnlineTradingEnv(gym.Env):
    def __init__(self, initial_balance, kis_client):
        super().__init__()
        self.kis_client = kis_client
        self.initial_balance = initial_balance
        self.current_step = 0
        self.buy_fee_rate = 0.0025
        self.sell_fee_rate = 0.0050
        self.sec_fee_rate = 0.0000278
        self.min_sec_fee = 0.01
        self._initialize_holdings()
        self.history = {
            'portfolio_value': [self.initial_balance],
            'asset_ratios': [{'soxl': 0.33, 'soxs': 0.33, 'cash': 0.34}]
        }

    def _initialize_holdings(self):
        p1 = max(self.kis_client.get_price("SOXL"), 1.0)
        p2 = max(self.kis_client.get_price("SOXS"), 1.0)
        allocation = {'soxl': 0.33, 'soxs': 0.33, 'cash': 0.34}
        self.holdings = {
            'soxl': (self.initial_balance * allocation['soxl']) / p1,
            'soxs': (self.initial_balance * allocation['soxs']) / p2,
            'cash': self.initial_balance * allocation['cash']
        }
        self.last_portfolio_value = self.initial_balance

    def _get_obs(self):
        p1 = max(self.kis_client.get_price("SOXL"), 1.0)
        p2 = max(self.kis_client.get_price("SOXS"), 1.0)
        # 오프라인 코드와 동일한 5차원 구조
        return np.array([p1, p2, self.holdings['soxl'], self.holdings['soxs'], self.holdings['cash']], dtype=np.float32)

    def step(self, action):
        p1 = max(self.kis_client.get_price("SOXL"), 1.0)
        p2 = max(self.kis_client.get_price("SOXS"), 1.0)

        if action == 1 and self.holdings['cash'] > 0:
            cash = self.holdings['cash']
            qty = int(cash / p1)
            if self.kis_client.place_order("SOXL", "buy", qty):
                self.holdings['soxl'] += qty * (1 - self.buy_fee_rate)
                self.holdings['cash'] = 0
        elif action == 2 and self.holdings['cash'] > 0:
            cash = self.holdings['cash']
            qty = int(cash / p2)
            if self.kis_client.place_order("SOXS", "buy", qty):
                self.holdings['soxs'] += qty * (1 - self.buy_fee_rate)
                self.holdings['cash'] = 0
        elif action == 3 and self.holdings['soxl'] > 0:
            qty = int(self.holdings['soxl'])
            if self.kis_client.place_order("SOXL", "sell", qty):
                val = qty * p1
                fee = max(val * self.sec_fee_rate, self.min_sec_fee)
                self.holdings['cash'] += val * (1 - self.sell_fee_rate) - fee
                self.holdings['soxl'] = 0
        elif action == 4 and self.holdings['soxs'] > 0:
            qty = int(self.holdings['soxs'])
            if self.kis_client.place_order("SOXS", "sell", qty):
                val = qty * p2
                fee = max(val * self.sec_fee_rate, self.min_sec_fee)
                self.holdings['cash'] += val * (1 - self.sell_fee_rate) - fee
                self.holdings['soxs'] = 0

        self.current_step += 1
        total_value = self.holdings['soxl'] * p1 + self.holdings['soxs'] * p2 + self.holdings['cash']
        self.last_portfolio_value = total_value
        allocation = {
            'soxl': self.holdings['soxl'] * p1 / total_value if total_value > 0 else 0,
            'soxs': self.holdings['soxs'] * p2 / total_value if total_value > 0 else 0,
            'cash': self.holdings['cash'] / total_value if total_value > 0 else 1
        }
        self.history['portfolio_value'].append(total_value)
        self.history['asset_ratios'].append(allocation)
        reward = (total_value - self.initial_balance) / self.initial_balance
        done = False
        return self._get_obs(), reward, done, False, {'portfolio_value': total_value}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self._initialize_holdings()
        self.history = {
            'portfolio_value': [self.initial_balance],
            'asset_ratios': [{'soxl': 0.33, 'soxs': 0.33, 'cash': 0.34}]
        }
        return self._get_obs(), {}

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

    @property
    def action_space(self):
        return spaces.Discrete(5)

# === 관측값 패딩 래퍼 (5차원을 56차원으로 확장) ===
class ObservationPaddingWrapper(gym.ObservationWrapper):
    def __init__(self, env, target_dim=56):
        super().__init__(env)
        self.target_dim = target_dim
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(target_dim,), dtype=np.float32)

    def observation(self, obs):
        current_dim = obs.shape[0]
        if current_dim < self.target_dim:
            pad_width = self.target_dim - current_dim
            obs = np.pad(obs, (0, pad_width), mode='constant', constant_values=0)
        return obs

# === 실시간 모니터링 ===
class RealTimeMonitor:
    def __init__(self):
        self.enabled = True
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 10))
        plt.subplots_adjust(hspace=0.2)
        plt.ion()
        plt.show(block=False)
        self.update_count = 0

    def update(self, history):
        if not self.enabled or not history['portfolio_value']:
            return

        portfolio_values = [v for v in history['portfolio_value'] if not np.isnan(v)]
        steps = range(len(portfolio_values))
        if len(portfolio_values) <= 1:
            print("[Monitor] Insufficient data:", len(portfolio_values), "points")
            return

        soxl_ratios = [r['soxl'] for r in history['asset_ratios'][:len(portfolio_values)]]
        soxs_ratios = [r['soxs'] for r in history['asset_ratios'][:len(portfolio_values)]]
        cash_ratios = [r['cash'] for r in history['asset_ratios'][:len(portfolio_values)]]

        self.ax1.clear()
        self.ax1.plot(steps, portfolio_values, 'b-', label='Portfolio Value')
        self.ax1.set_title('Portfolio Value')
        self.ax1.legend()

        self.ax2.clear()
        self.ax2.plot(steps, soxl_ratios, label='SOXL')
        self.ax2.plot(steps, soxs_ratios, label='SOXS')
        self.ax2.plot(steps, cash_ratios, label='Cash')
        self.ax2.set_ylim(0, 1)
        self.ax2.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

        self.update_count += 1
        if self.update_count % 10 == 0:
            self.save(f"monitoring/online_update_{self.update_count}.png")

    def save(self, filename):
        try:
            plt.ioff()
            abs_path = os.path.abspath(filename)
            self.fig.savefig(abs_path)
            plt.ion()
            print("[Monitor] 그래프 저장됨:", abs_path)
        except Exception as e:
            print("[Monitor] 저장 실패:", e)

# === 메인 실행 로직 ===
if __name__ == "__main__":
    os.makedirs("monitoring", exist_ok=True)

    kis_client = KISClient()
    # 온라인 환경 생성 후 관측값 패딩 래퍼를 적용하여 56차원으로 확장
    base_env = OnlineTradingEnv(initial_balance=500000 / 1300.0, kis_client=kis_client)
    env = ObservationPaddingWrapper(base_env, target_dim=56)
    vec_env = DummyVecEnv([lambda: env])
    monitor = RealTimeMonitor()

    # 모델 파일 경로 지정
    # 모델 파일 경로 지정 (같은 폴더에 모델 파일이 있다고 가정)
    model_path = "offline_dqn_model_final.zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    print(f"[INFO] 모델 파일 로드 시도: {model_path}")

    # 모델 로드
    model = DQN.load(model_path, env=vec_env)

    # 학습 시 사용했던 56차원 관측값 구조에 맞게 모델을 로드
    model = DQN.load(model_path, env=vec_env)

    # 실시간 거래 루프: 미국 증시가 열려 있고, 밤 11시 ~ 새벽 5시 사이에만 실행
    obs = vec_env.reset()
    while True:
        now = datetime.datetime.now().time()
        if (now >= datetime.time(23, 0) or now < datetime.time(5, 0)) and is_us_market_open():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            monitor.update(base_env.history)
            print(f"Step: {base_env.current_step}, Action: {action}, Reward: {reward}, Portfolio: {info[0]['portfolio_value']}")
            time.sleep(60)  # 1분 간격 실행
        else:
            print("현재 실행 시간 외이거나 미국 증시가 휴장입니다. 대기 중...")
            time.sleep(300)  # 실행 시간 외에는 5분마다 체크
