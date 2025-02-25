import requests
import json
import time
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from solana.rpc.api import Client
from solana.keypair import Keypair
from solana.transaction import Transaction
from solana.system_program import TransferParams, transfer
from solders.pubkey import Pubkey
from solders.compute_budget import set_compute_unit_price
import logging
import tweepy
from spl.token.client import Token
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import create_associated_token_account, get_associated_token_address

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PumpFunTradingBot:
    def __init__(self, x_api_key, x_api_secret, x_access_token, x_access_secret, solsniffer_api_key, wallet_private_key):
        # Solana client
        self.solana_client = Client("https://api.mainnet-beta.solana.com")
        self.pump_fun_api = "https://api.pump.fun/trades"
        self.amm_api = "https://amm.pump.fun/pools"
        
        # X API setup
        auth = tweepy.OAuthHandler(x_api_key, x_api_secret)
        auth.set_access_token(x_access_token, x_access_secret)
        self.x_api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Solsniffer API
        self.solsniffer_api = "https://solsniffer.com/api/v1/scan"
        self.solsniffer_api_key = solsniffer_api_key
        
        # Wallet setup
        self.wallet = Keypair.from_base58_string(wallet_private_key)
        self.token_client = Token(self.solana_client, Pubkey(TOKEN_PROGRAM_ID), self.wallet)
        
        # Trading parameters
        self.priority_fee_lamports = 100000  # 0.0001 SOL priority fee
        self.buy_amount_sol = 1 * 10**9  # 1 SOL in lamports
        self.slippage_percent = 15  # 15% slippage
        self.take_profit_multiplier = 10  # 10x take-profit
        self.moonbag_percent = 15  # 15% moonbag
        
        # Analysis parameters
        self.min_liquidity = 1000  # Minimum SOL liquidity
        self.min_volume_24h = 5000  # Minimum 24h volume in SOL
        self.min_snifscore = 85  # Minimum Solsniffer score
        self.notable_follower_threshold = 10000  # Notable follower threshold
        self.max_username_changes = 3  # Max username changes flag
        
        # Known ruggers/cabals (placeholder - replace with real data)
        self.known_ruggers = set([
            "rUgGeR12345678901234567890123456789012345",
            "CaBaL98765432109876543210987654321098765"
        ])
        
        # Scoring setup
        self.scaler = MinMaxScaler()
        self.weights = {
            'volume': 0.25,
            'liquidity': 0.25,
            'price_change': 0.20,
            'social_trust': 0.30
        }
        
        # Portfolio tracking
        self.positions = {}  # {token_address: {'amount': int, 'buy_price': float, 'buy_time': datetime}}

    def fetch_pump_fun_data(self):
        """Fetch trading data from Pump.fun"""
        try:
            response = requests.get(self.pump_fun_api, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching Pump.fun data: {e}")
            return []

    def fetch_amm_data(self):
        """Fetch AMM pool data"""
        try:
            response = requests.get(self.amm_api, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching AMM data: {e}")
            return []

    def fetch_solsniffer_score(self, token_address):
        """Fetch security score from Solsniffer API"""
        try:
            headers = {"Authorization": f"Bearer {self.solsniffer_api_key}"}
            params = {"contract_address": token_address}
            response = requests.get(self.solsniffer_api, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return {
                'snifscore': data.get('snifscore', 0),
                'risks': data.get('security_indicators', []),
                'deployer': data.get('deployer_address', '')
            }
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching Solsniffer data for {token_address}: {e}")
            return {'snifscore': 0, 'risks': [], 'deployer': ''}

    def get_recent_tx_count(self, token_address):
        """Get recent transaction count from Solana blockchain"""
        try:
            signatures = self.solana_client.get_signatures_for_address(Pubkey.from_string(token_address), limit=100)
            return len(signatures['result'])
        except Exception as e:
            logging.error(f"Error fetching tx count for {token_address}: {e}")
            return 0

    def detect_fake_volume(self, token_data, snifscore_data):
        """Detect potential fake volume patterns"""
        volume_24h = token_data.get('volume_24h', 0)
        transaction_count = self.get_recent_tx_count(token_data['mint_address'])
        volume_tx_ratio = volume_24h / max(transaction_count, 1)
        return volume_tx_ratio > 1000 or any(risk.get('type') == 'wash_trading' for risk in snifscore_data['risks'])

    def check_rugger_cabal(self, deployer_address):
        """Check if token deployer is a known rugger or cabal"""
        return deployer_address in self.known_ruggers

    def analyze_x_profile(self, x_handle):
        """Analyze project's X profile for trust metrics"""
        try:
            user = self.x_api.get_user(screen_name=x_handle)
            notable_followers = sum(1 for follower in self.x_api.get_followers(screen_name=x_handle, count=100)
                                  if follower.followers_count >= self.notable_follower_threshold)
            trust_score = sum([
                0.3 if user.verified else 0,
                min((datetime.now().year - user.created_at.year) / 5, 0.3),
                min(user.followers_count / max(user.friends_count, 1), 0.2),
                min(notable_followers / 5, 0.2)
            ])
            username_change_flag = user.followers_count < 1000 and (datetime.now().year - user.created_at.year) < 1
            return {
                'trust_score': trust_score,
                'notable_followers': notable_followers,
                'username_change_flag': username_change_flag,
                'raw_followers': user.followers_count
            }
        except tweepy.TweepError as e:
            logging.error(f"Error analyzing X profile {x_handle}: {e}")
            return {'trust_score': 0, 'notable_followers': 0, 'username_change_flag': False, 'raw_followers': 0}

    def get_current_price(self, token_address):
        """Fetch current price from AMM pool data"""
        try:
            amm_data = self.fetch_amm_data()
            for pool in amm_data:
                if pool['mint_address'] == token_address:
                    return pool.get('price_sol', 0)
            return 0
        except Exception as e:
            logging.error(f"Error fetching price for {token_address}: {e}")
            return 0

    def buy_token(self, token_address, symbol):
        """Execute buy transaction with slippage and priority fee"""
        try:
            token_pubkey = Pubkey.from_string(token_address)
            price = self.get_current_price(token_address)
            if not price:
                logging.error(f"No price available for {symbol}")
                return False

            expected_tokens = (self.buy_amount_sol / price) * (1 - self.slippage_percent / 100)
            lamports_with_fee = self.buy_amount_sol + self.priority_fee_lamports

            # Check balance
            balance = self.solana_client.get_balance(self.wallet.public_key)['result']['value']
            if balance < lamports_with_fee:
                logging.error(f"Insufficient balance for {symbol}: {balance/10**9} SOL available")
                return False

            # Create associated token account if needed
            ata = get_associated_token_address(self.wallet.public_key, token_pubkey)
            if not self.solana_client.get_account_info(ata)['result']['value']:
                tx = Transaction().add(
                    create_associated_token_account(self.wallet.public_key, self.wallet.public_key, token_pubkey)
                )
                self.solana_client.send_transaction(tx, self.wallet)

            # Simplified buy transaction (replace with actual Pump.fun AMM swap)
            tx = Transaction().add(
                set_compute_unit_price(self.priority_fee_lamports // 1000),  # Micro-lamports
                transfer(TransferParams(
                    from_pubkey=self.wallet.public_key,
                    to_pubkey=Pubkey.from_string("PumpFunAMMAddress"),  # Placeholder
                    lamports=self.buy_amount_sol
                ))
            )
            response = self.solana_client.send_transaction(tx, self.wallet)
            tx_hash = response['result']
            logging.info(f"Buy {symbol}: {self.buy_amount_sol/10**9} SOL for ~{expected_tokens:.2f} tokens, tx: {tx_hash}")

            self.positions[token_address] = {
                'amount': expected_tokens,
                'buy_price': price,
                'buy_time': datetime.now(),
                'symbol': symbol
            }
            return True
        except Exception as e:
            logging.error(f"Buy failed for {symbol}: {e}")
            return False

    def sell_token(self, token_address, symbol, sell_all=False):
        """Execute sell transaction with take-profit and moonbag"""
        try:
            if token_address not in self.positions:
                return False

            position = self.positions[token_address]
            current_price = self.get_current_price(token_address)
            if not current_price:
                return False

            profit_ratio = current_price / position['buy_price']
            total_amount = position['amount']

            if profit_ratio >= self.take_profit_multiplier or sell_all:
                moonbag_amount = total_amount * (self.moonbag_percent / 100) if not sell_all else 0
                sell_amount = total_amount - moonbag_amount
                min_sol_expected = (sell_amount * current_price) * (1 - self.slippage_percent / 100)

                token_pubkey = Pubkey.from_string(token_address)
                ata = get_associated_token_address(self.wallet.public_key, token_pubkey)
                
                tx = Transaction().add(
                    set_compute_unit_price(self.priority_fee_lamports // 1000),
                    self.token_client.transfer(
                        source=ata,
                        dest=Pubkey.from_string("PumpFunAMMAddress"),  # Placeholder
                        owner=self.wallet.public_key,
                        amount=int(sell_amount)
                    )
                )
                response = self.solana_client.send_transaction(tx, self.wallet)
                tx_hash = response['result']
                logging.info(f"Sell {symbol}: {sell_amount:.2f} tokens for ~{min_sol_expected/10**9:.4f} SOL, tx: {tx_hash}")

                if sell_all or moonbag_amount == 0:
                    del self.positions[token_address]
                else:
                    self.positions[token_address]['amount'] = moonbag_amount
                return True
            return False
        except Exception as e:
            logging.error(f"Sell failed for {symbol}: {e}")
            return False

    def analyze_token(self, token_data):
        """Analyze token metrics and return profitability score"""
        try:
            volume_24h = token_data.get('volume_24h', 0)
            liquidity = token_data.get('liquidity', 0)
            price_change_24h = token_data.get('price_change_24h', 0)
            x_handle = token_data.get('x_handle', '')
            token_address = token_data['mint_address']
            symbol = token_data.get('symbol', 'Unknown')

            # Initial filters
            if liquidity < self.min_liquidity or volume_24h < self.min_volume_24h:
                return None

            # Solsniffer check
            snifscore_data = self.fetch_solsniffer_score(token_address)
            if snifscore_data['snifscore'] < self.min_snifscore:
                logging.info(f"{symbol} rejected: Snifscore {snifscore_data['snifscore']} < {self.min_snifscore}")
                return None

            # Fake volume and rugger checks
            if self.detect_fake_volume(token_data, snifscore_data):
                logging.warning(f"{symbol} flagged for potential fake volume")
                return None
            if self.check_rugger_cabal(snifscore_data['deployer']):
                logging.warning(f"{symbol} flagged: Deployer in known rugger/cabal list")
                return None

            # X profile analysis
            social_metrics = self.analyze_x_profile(x_handle) if x_handle else {
                'trust_score': 0, 'notable_followers': 0, 'username_change_flag': False, 'raw_followers': 0
            }
            if social_metrics['username_change_flag']:
                logging.warning(f"{symbol} flagged for suspicious username changes")

            # Calculate score
            metrics = np.array([[volume_24h, liquidity, price_change_24h, social_metrics['trust_score']]])
            normalized_metrics = self.scaler.fit_transform(metrics)[0]
            score = sum(normalized_metrics[i] * w for i, w in enumerate(self.weights.values()))
            if social_metrics['username_change_flag']:
                score *= 0.5

            return {
                'mint_address': token_address,
                'symbol': symbol,
                'score': score,
                'volume_24h': volume_24h,
                'liquidity': liquidity,
                'price_change_24h': price_change_24h,
                'social_trust': social_metrics['trust_score'],
                'notable_followers': social_metrics['notable_followers'],
                'username_change_flag': social_metrics['username_change_flag'],
                'snifscore': snifscore_data['snifscore']
            }
        except Exception as e:
            logging.error(f"Error analyzing token {token_data.get('symbol', 'Unknown')}: {e}")
            return None

    def manage_positions(self):
        """Monitor and manage existing positions"""
        for token_address in list(self.positions.keys()):
            position = self.positions[token_address]
            self.sell_token(token_address, position['symbol'])

    def run(self):
        """Main bot execution loop"""
        logging.info("Starting Pump.fun Trading Bot...")
        
        while True:
            try:
                pump_trades = self.fetch_pump_fun_data()
                amm_pools = self.fetch_amm_data()

                all_tokens = {}
                for trade in pump_trades:
                    all_tokens[trade['mint_address']] = trade
                for pool in amm_pools:
                    all_tokens[pool['mint_address']] = {**all_tokens.get(pool['mint_address'], {}), **pool}

                opportunities = []
                for token_data in all_tokens.values():
                    if token_data['mint_address'] not in self.positions:
                        analysis = self.analyze_token(token_data)
                        if analysis:
                            opportunities.append(analysis)

                opportunities.sort(key=lambda x: x['score'], reverse=True)
                
                # Trading logic
                if opportunities and opportunities[0]['score'] > 0.7:
                    top_opp = opportunities[0]
                    self.buy_token(top_opp['mint_address'], top_opp['symbol'])

                self.manage_positions()

                # Display results
                logging.info(f"\nTop Trading Opportunities ({datetime.now()}):")
                for opp in opportunities[:5]:
                    logging.info(f"Token: {opp['symbol']} ({opp['mint_address']})")
                    logging.info(f"Score: {opp['score']:.3f}")
                    logging.info(f"Snifscore: {opp['snifscore']}")
                    logging.info(f"24h Volume: {opp['volume_24h']} SOL")
                    logging.info(f"Liquidity: {opp['liquidity']} SOL")
                    logging.info(f"24h Price Change: {opp['price_change_24h']}%")
                    logging.info(f"Social Trust: {opp['social_trust']:.3f}")
                    logging.info(f"Notable Followers: {opp['notable_followers']}")
                    if opp['username_change_flag']:
                        logging.warning("WARNING: Suspicious username change history")
                    logging.info("------------------------")

                logging.info(f"Current Positions: {len(self.positions)}")
                for addr, pos in self.positions.items():
                    logging.info(f"{pos['symbol']}: {pos['amount']:.2f} @ {pos['buy_price']:.6f} SOL")

                time.sleep(300)  # 5-minute interval

            except Exception as e:
                logging.error(f"Main loop error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    # Configuration (replace with your credentials)
    config = {
        "x_api_key": "66HYkH0nOgOggwmsoAYuVc5nt"
        "x_api_secret": "nwqh4zYtMRL1hIC0JBlzyVuKvsvjp3CFo8qeow6C5hHNzk3g8b",
        "x_access_token": "2298165384-p3KITvBSS3I0WH2Cx7ML1F1Dz6gKQKKGhASfQlp",
        "x_access_secret": "fIyJOGdkPZaJbhEyGy4PvSp54IKasGSXqddkKljAG5PWD",
        "solsniffer_api_key": "8ir9qhy4h2ig336570sbby5ujotfgf",
        "wallet_private_key": "2zXXp5U88G21MG62SfnvDFH7LBTeua63g2EhtgWfi2PXVwPibypp7LRwYXfxgv3PiaVuy8xK2BnmwSZoaG3wcdj2",
            }
    
    
    bot = PumpFunTradingBot(**config)
    bot.run()
