from typing import Tuple
import gym
from gym import spaces
import pandas as pd
import numpy as np
from typing import Any, Sequence
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PositionSide(object):
    NONE: int = 0
    BUY: int = 1
    SELL: int = 2

class PositionStatus(object):
    NONE: int = 0
    OPEN: int = 1
    CLOSE: int = 2
    CANCLE: int = 3

class Position(object):
    start_bar_idx: int = 0
    end_bar_idx: int = -1
    price: np.float64 = 0
    vol: int = 0
    limit_price: np.float64 = 0
    stop_price: np.float64 = 0
    margin: np.float64 = 0
    commission: np.float64 = 0
    side: PositionSide = PositionSide.NONE
    status: PositionStatus = PositionStatus.NONE

    def __init__(self, 
                 start_bar_idx: int = 0, 
                 end_bar_idx: int = -1, 
                 price: np.float64 = 0, 
                 vol: int = 0, 
                 limit_price: np.float64 = 0, 
                 stop_price: np.float64 = 0, 
                 margin: np.float64 = 0,
                 commission: np.float64 = 0,
                 side: PositionSide = PositionSide.NONE,
                 status: PositionStatus = PositionStatus.NONE):
        self.start_bar_idx = start_bar_idx
        self.end_bar_idx = end_bar_idx
        self.price = price
        self.vol = vol
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.margin = margin
        self.commission = commission
        self.side = side
        self.status = status

    def __repr__(self) -> str:
        return f'position(start_bar_idx={self.start_bar_idx}, end_bar_idx={self.end_bar_idx}, price={self.price}, vol={self.vol}, limit_price={self.limit_price}, stop_price={self.stop_price}, margin={self.margin}, commission={self.commission}, side={self.side}, status={self.status})'

class BarEnvV1(gym.Env):

    current_idx: int = 0
    initial_cash: np.float64 = 100
    cash: np.float64 = 0
    commission_rate: np.float64 = 0
    leverage: int = 1
    value: np.float64 = 0
    positions: Sequence[Position] = []
    actions: Sequence = []
    _terminated: bool = False
    _truncate: bool = False

    def __init__(self, df: pd.DataFrame, window_size: int = 20):
        super(BarEnvV1, self).__init__()
        self.df = df
        self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Box(low=0, high=1e8, shape=(3,), dtype=np.float32), ))
        self.observation_space = spaces.Box(low=0, high=1, shape=(window_size, len(df.columns)), dtype=pd.DataFrame)
        self.window_size = window_size
        self.current_idx = 0
    
    def _calculate_trade_margin(self, vol: int, price: np.float64) -> np.float64:
        return (price * vol) / self.leverage
    
    def _calculate_commissions(self, vol: int, price: np.float64) -> np.float64:
        return (price * vol) * self.commission_rate

    def _calculate_last_reward(self, base_line_rate: np.float64 = 0.1) -> np.float64:
        if (self.value - self.initial_cash) / self.initial_cash < base_line_rate:
            return -10 + self.value - self.initial_cash
        
    def _get_trade_info(self) -> dict:
        return {
            'cash': self.cash,
            'commission_rate': self.commission_rate,
            'leverage': self.leverage,
            'value': self.value,
            'positions': self.positions
        }
    
    def _get_open_position_profit(self) -> np.float64:
        profit: np.float64 = 0
        for pos in self.positions:
            if pos.status != PositionStatus.OPEN:
                continue
            if pos.side == PositionSide.BUY:
                profit += pos.vol * (self.df.iloc[self.current_idx - 1]['close'] - pos.price)
            elif pos.side == PositionSide.SELL:
                profit += pos.vol * (pos.price - self.df.iloc[self.current_idx - 1]['close'])
        return profit
    
    def _get_position_margins(self) -> np.float64:
        margin: np.float64 = 0
        for pos in self.positions:
            if pos.status != PositionStatus.OPEN:
                continue
            margin += pos.margin
        return margin
    
    def _update_values(self) -> None:
        self.value = self.cash
        self.value += self._get_open_position_profit()
        self.value += self._get_position_margins()

    def _close_position(self, pos: Position, price: np.float64) -> np.float64:
        commission = self._calculate_commissions(pos.vol, price)
        profit: np.float64 = 0
        if pos.side == PositionSide.BUY:
            profit += pos.vol * (price - pos.price)
        elif pos.side == PositionSide.SELL:
            profit += pos.vol * (pos.price - price)
        pos.status = PositionStatus.CLOSE
        pos.end_bar_idx = self.current_idx
        pos.commission += commission
        profit -= pos.commission
        self.cash += pos.margin + profit
        return profit


    def _update_positions(self) -> np.float64:
        total_profit: np.float64 = 0
        for pos in self.positions:
            if pos.status != PositionStatus.OPEN:
                continue

            if pos.side == PositionSide.BUY:
                if pos.stop_price > 0 and self.df.iloc[self.current_idx - 1]['low'] <= pos.stop_price:
                    total_profit += self._close_position(pos, self.df.iloc[self.current_idx - 1]['low'])
                    continue
                if pos.limit_price > 0 and self.df.iloc[self.current_idx - 1]['high'] > pos.limit_price:
                    total_profit += self._close_position(pos, pos.limit_price)
                    continue
            elif pos.side == PositionSide.SELL:
                if pos.stop_price > 0 and self.df.iloc[self.current_idx - 1]['high'] >= pos.stop_price:
                    total_profit += self._close_position(pos, self.df.iloc[self.current_idx - 1]['high'])
                    continue
                if pos.limit_price > 0 and self.df.iloc[self.current_idx - 1]['low'] < pos.limit_price:
                    total_profit += self._close_position(pos, pos.limit_price)
                    continue
        return total_profit
    
    def _get_next_observation(self) -> Tuple[pd.DataFrame, bool, bool]:
        if self.current_idx >= self.df.shape[0] - 1:
            return self.df.iloc[-self.window_size:], True, True
        else:
            return self.df.iloc[self.current_idx - self.window_size: self.current_idx], False, False

    def _is_need_to_terminate(self, dt: datetime) -> bool:
        return (dt.hour == 7 and dt.minute == 55) or (dt.hour == 15 and dt.minute == 55) or (dt.hour == 23 and dt.minute == 55)
    def step(self, action: dict) -> Tuple[Any, np.float64, bool, bool, dict]:
        if self._terminated:
            raise RuntimeError("Episode is terminated")
        
        self.actions.append(action)

        # 1. 更新current_idx 并检查是否越界
        self.current_idx += 1
        self._terminated = False
        self._truncated = False
        reward: np.float64 = 0
        info = {}
        ob_prime, self._truncated, self._terminated = self._get_next_observation()
        dt: datetime = self.df.iloc[self.current_idx - 1]['datetime']

        side: int = action['side']
        prices: Tuple[np.float64, np.float64, np.float64] = action['prices']
        vol: int = action['vol']

        # 2. 判断当前输入价格是否额能够成交
        if side == 0:
            pass
        elif side == 1:
            # buy
            if prices[0] > ob_prime.iloc[-1]['low']:
                # 成交
                reward += self.deal_reward
                margin = self._calculate_trade_margin(vol, prices[0])
                commission_amount = self._calculate_commissions(vol, prices[0])
                self.cash -= (margin)
                self.positions.append(Position(
                    start_bar_idx=self.current_idx, 
                    end_bar_idx=-1, 
                    price=prices[0], 
                    limit_price=prices[1], 
                    stop_price=prices[2], 
                    vol=vol, margin=margin, 
                    commission=commission_amount, 
                    side=PositionSide.BUY, 
                    status=PositionStatus.OPEN))
        elif side == 2:
            # sell
            if prices[0] < ob_prime.iloc[-1]['high']:
                # 成交
                reward += self.deal_reward
                margin = self._calculate_trade_margin(vol, prices[0])
                commission_amount = self._calculate_commissions(vol, prices[0])
                self.cash -= (margin)
                self.positions.append(Position(
                    start_bar_idx=self.current_idx, 
                    end_bar_idx=-1, 
                    price=prices[0], 
                    limit_price=prices[1], 
                    stop_price=prices[2], 
                    vol=vol, 
                    margin=margin, 
                    commission=commission_amount, 
                    side=PositionSide.SELL, 
                    status=PositionStatus.OPEN))
                logger.debug(f'sell {vol} at {prices[0]}, positions: {self.positions[-1]}, commission_amount: {commission_amount}')
        
        # 3. 更新持仓
        total_profit = self._update_positions()
        

        # 4. 判断是否需要终止
        if self._is_need_to_terminate(dt):
            for pos in self.positions:
                if pos.status != PositionStatus.OPEN:
                    continue
                total_profit += self._close_position(pos, ob_prime.iloc[-1]['close'])
            self._terminated = True
            self._truncated = True

        # 5. 计算当前持仓的价值
        self._update_values()

        # 6. 计算奖励
        if self.value < self.initial_cash * 0.8:
            reward -= 20
            self._terminated = True
            self._truncated = True
        else:
            reward += total_profit

        return ob_prime, reward, self._terminated, self._truncated, self._get_trade_info()
    
    def reset(self, cash: np.float64 = 100, commission_rate: np.float64 = 5e-4, leverage: int = 20, deal_reward = 0.1) -> Any:
        """
        重置环境
        
        :param cash: 初始资金
        :param commission_rate: 交易手续费率
        :param leverage: 杠杆
        :param deal_reward: 成交奖励
        """
        self.initial_cash = cash
        self.cash = cash
        self.commission_rate = commission_rate
        self.leverage = leverage
        self.value = cash
        self.deal_reward = deal_reward
        self.positions = []
        self.actions = []
        if self.current_idx == 0:
            self.current_idx = self.window_size
        else:
            self.current_idx += 1
        
        if self.current_idx >= self.df.shape[0]:
            self.current_idx = self.window_size
        
        return self.df.iloc[self.current_idx - self.window_size: self.current_idx]
