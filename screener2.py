import os
import json
import requests
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import argparse
from tabulate import tabulate
from itertools import combinations, product
from dotenv import load_dotenv
from auth import get_access_token
import sched
import time
import threading
import multiprocessing
import logging
import datetime
import itertools
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from collections import namedtuple

# Set up logging
log_filename = f"logs/options_strategy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add a stream handler to also print to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Only INFO and above will be printed to console
logging.getLogger().addHandler(console_handler)

# Function to log and print
def log_print(message, level=logging.INFO):
    logging.log(level, message)
    if level >= logging.INFO:
        print(message)


class SchwabOptionsDataFetcher:
    def __init__(self):
        load_dotenv()
        self.MARKET_BASE_URL = os.getenv('MARKET_BASE_URL')
        self.access_token = None
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.refresh_token()
        self.schedule_token_refresh()

    def schedule_token_refresh(self):
        self.scheduler.enter(3540, 1, self.refresh_token)  # 59 minutes
        threading.Thread(target=self.scheduler.run, daemon=True).start()

    def refresh_token(self):
        try:
            self.access_token = get_access_token()
            if not self.access_token:
                raise ValueError("Failed to retrieve access token")
            print("Access token refreshed successfully.")
        except Exception as e:
            print(f"Error refreshing access token: {str(e)}")
            print("Please check your .env file and auth.py implementation.")
            self.access_token = None

    def get_options_chain(self, symbol, fromDate, toDate, ranges="OTM", contract_type="CALL",):
        if not self.MARKET_BASE_URL:
            raise ValueError("MARKET_BASE_URL is not set in .env file")
        if not self.access_token:
            raise ValueError("Access token is not available. Check your authentication setup.")
        
        params = {
            'symbol': symbol,
            'fromDate': fromDate,
            'toDate': toDate,
            'contractType': contract_type,
            # 'strikeCount': 50,
            'includeUnderlyingQuote': 'true',
            'strategy': 'ANALYTICAL',
            'range': ranges
        }
        
        try:
            response = requests.get(f'{self.MARKET_BASE_URL}/chains', headers={'Authorization': f'Bearer {self.access_token}'}, params=params)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            data = response.json()
            
            # Debug: Print the keys in the response
            print("Response keys:", data.keys())
            
            # Debug: Print a sample of the data structure
            if 'callExpDateMap' in data:
                print("Sample call option data structure:")
                sample_date = next(iter(data['callExpDateMap']))
                sample_strike = next(iter(data['callExpDateMap'][sample_date]))
                print(json.dumps(data['callExpDateMap'][sample_date][sample_strike][0], indent=2))
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching options chain: {str(e)}")
            print(f"Response status code: {response.status_code}")
            print(f"Response content: {response.text}")
            raise

    def extract_option_data(self, json_data):
        options_data = []
    
        for exp_map_key in ['callExpDateMap', 'putExpDateMap']:
            if exp_map_key in json_data:
                for date, strikes in json_data[exp_map_key].items():
                    for strike, options in strikes.items():
                        for option in options:
                            options_data.append({
                                'symbol': option.get('symbol'),
                                'type': 'call' if exp_map_key == 'callExpDateMap' else 'put',
                                'strike': float(strike),
                                'price': option.get('last', 0),
                                'delta': option.get('delta', 0),
                                'gamma': option.get('gamma', 0),
                                'theta': option.get('theta', 0),
                                'vega': option.get('vega', 0),
                                'rho': option.get('rho', 0),
                                'askPrice': option.get('ask', 0),
                                'bidPrice': option.get('bid', 0),
                                'openInterest': option.get('openInterest', 0),
                                'totalVolume': option.get('totalVolume', 0),
                                'impliedVolatility': option.get('volatility', 0),
                                'inTheMoney': option.get('inTheMoney', False),
                                'expirationDate': option.get('expirationDate', ''),
                                'daysToExpiration': option.get('daysToExpiration', 0)
                            })
        print(pd.DataFrame(options_data))
        return pd.DataFrame(options_data)
    
    def get_stock_price(self, json_data):
        return json_data.get('underlyingPrice', 0)

# Define a flexible Option namedtuple
Option = namedtuple('Option', ['symbol', 'type', 'strike', 'price', 'delta', 'gamma', 'vega', 'theta', 'rho', 'expirationDate'])
Option.__new__.__defaults__ = (None,) * len(Option._fields)

class OptionsStrategyAnalyzer:
    def __init__(self, options_data):
        self.options_data = [Option(**{k: v for k, v in option.items() if k in Option._fields}) for option in options_data]
    
    def calculate_credit_debit(self, legs):
        return sum(-leg['direction'] * leg['option'].price for leg in legs)

    def create_strategy_identifier(self, legs):
        # Sort legs by strike price and option type to ensure consistent ordering
        sorted_legs = sorted(legs, key=lambda x: (x['option'].strike, x['option'].type))
        return tuple((leg['option'].type, leg['option'].strike, leg['direction']) for leg in sorted_legs)
    
    def create_strategies(self, max_legs=3):
        strategies = {}
        for n in range(1, max_legs + 1):
            for combo in itertools.combinations(self.options_data, n):
                for direction in itertools.product([1, -1], repeat=n):
                    legs = [{"option": leg, "direction": d} for leg, d in zip(combo, direction)]
                    strategy_id = self.create_strategy_identifier(legs)

                    if strategy_id not in strategies:
                        strategy_type = self.identify_strategy(legs)
                        max_loss, max_profit = self.calculate_max_loss_profit(legs, strategy_type)
                        credit_debit = self.calculate_credit_debit(legs)
                        strategy = {
                            'legs': legs,
                            'strategy_type': strategy_type,
                            'max_loss': max_loss,
                            'max_profit': max_profit,
                            'credit_debit': credit_debit,
                            'net_gamma': sum(leg['direction'] * leg['option'].gamma for leg in legs),
                            'net_vega': sum(leg['direction'] * leg['option'].vega for leg in legs),
                            'net_theta': sum(leg['direction'] * leg['option'].theta for leg in legs)
                        }
                        strategies[strategy_id] = strategy

        return list(strategies.values())

    def identify_strategy(self, legs):
        if len(legs) == 1:
            return f"{'Short' if legs[0]['direction'] == -1 else 'Long'} {legs[0]['option'].type.capitalize()}"
        elif len(legs) == 2:
            if legs[0]['option'].type != legs[1]['option'].type:
                return "Straddle" if legs[0]['option'].strike == legs[1]['option'].strike else "Strangle"
            elif legs[0]['direction'] != legs[1]['direction']:
                return f"{'Call' if legs[0]['option'].type == 'call' else 'Put'} Vertical Spread"
        elif len(legs) == 4:
            if len(set(leg['option'].strike for leg in legs)) == 3:
                return "Butterfly"
            elif len(set(leg['option'].strike for leg in legs)) == 2:
                return "Iron Condor"
        return "Custom"

    def calculate_max_loss_profit(self, legs, strategy_type):
        if strategy_type in ["Long Call", "Long Put"]:
            option = legs[0]['option']
            max_loss = option.price
            if strategy_type == "Long Call":
                max_profit = float('inf')
            else:  # Long Put
                max_profit = option.strike - option.price

        elif strategy_type in ["Short Call", "Short Put"]:
            option = legs[0]['option']
            max_profit = option.price
            if strategy_type == "Short Call":
                max_loss = float('inf')
            else:  # Short Put
                max_loss = option.strike

        elif strategy_type in ["Call Vertical Spread", "Put Vertical Spread"]:
            lower_strike_leg = min(legs, key=lambda x: x['option'].strike)
            higher_strike_leg = max(legs, key=lambda x: x['option'].strike)
            width = higher_strike_leg['option'].strike - lower_strike_leg['option'].strike

            if strategy_type == "Call Vertical Spread":
                if lower_strike_leg['direction'] == 1:  # Long lower strike, short higher strike (debit spread)
                    debit = lower_strike_leg['option'].price - higher_strike_leg['option'].price
                    max_profit = width - debit
                    max_loss = debit
                else:  # Short lower strike, long higher strike (credit spread)
                    credit = lower_strike_leg['option'].price - higher_strike_leg['option'].price
                    max_profit = credit
                    max_loss = width - credit
            else:  # Put Vertical Spread
                if lower_strike_leg['direction'] == -1:  # Short lower strike, long higher strike (credit spread)
                    credit = higher_strike_leg['option'].price - lower_strike_leg['option'].price
                    max_profit = credit
                    max_loss = width - credit
                else:  # Long lower strike, short higher strike (debit spread)
                    debit = higher_strike_leg['option'].price - lower_strike_leg['option'].price
                    max_profit = width - debit
                    max_loss = debit

        elif strategy_type in ["Straddle", "Strangle"]:
            total_premium = sum(leg['option'].price for leg in legs)
            if legs[0]['direction'] == 1:  # Long straddle/strangle
                max_loss = total_premium
                max_profit = float('inf')
            else:  # Short straddle/strangle
                max_loss = float('inf')
                max_profit = total_premium

        elif strategy_type == "Butterfly":
            middle_strike = sorted(set(leg['option'].strike for leg in legs))[1]
            outer_strikes = [leg['option'].strike for leg in legs if leg['option'].strike != middle_strike]
            width = min(abs(middle_strike - outer_strike) for outer_strike in outer_strikes)

            if legs[0]['direction'] == 1:  # Long butterfly
                debit = sum(leg['direction'] * leg['option'].price for leg in legs)
                max_loss = debit
                max_profit = width - debit
            else:  # Short butterfly
                credit = -sum(leg['direction'] * leg['option'].price for leg in legs)
                max_profit = credit
                max_loss = width - credit

        elif strategy_type == "Iron Condor":
            strikes = sorted(set(leg['option'].strike for leg in legs))
            width = min(strikes[1] - strikes[0], strikes[3] - strikes[2])
            credit = sum(leg['direction'] * leg['option'].price for leg in legs)
            max_profit = credit
            max_loss = width - credit

        else:  # Custom strategy
            # For custom strategies, we'll use a simplified approach
            total_debit = sum(max(0, leg['direction'] * leg['option'].price) for leg in legs)
            total_credit = sum(max(0, -leg['direction'] * leg['option'].price) for leg in legs)

            if total_debit > total_credit:  # Net debit strategy
                max_loss = total_debit - total_credit
                max_profit = float('inf')  # Assume unlimited profit potential for custom strategies
            else:  # Net credit strategy
                max_profit = total_credit - total_debit
                max_loss = float('inf')  # Assume unlimited loss potential for custom strategies

        # Handle edge cases
        if max_loss == 0 and max_profit == 0:
            # If both are zero, it's likely due to rounding errors or an invalid strategy
            max_loss = 0.01  # Set a small non-zero value
        elif max_loss == 0:
            # If max_loss is zero but max_profit isn't, it's a "risk-free" strategy
            pass  # Keep max_loss as zero
        
        return max_loss, max_profit

    def filter_strategies(self, strategies, risk_profile):
        if risk_profile == 'limited':
            return [s for s in strategies if self.is_limited_risk(s) and self.is_limited_profit(s)]
        elif risk_profile == 'unlimited':
            return [s for s in strategies if self.is_limited_risk(s) and not self.is_limited_profit(s)]
        else:
            return strategies

    def is_limited_risk(self, strategy):
        return strategy['max_loss'] != float('inf')

    def is_limited_profit(self, strategy):
        return strategy['max_profit'] != float('inf')

    def optimize_strategies(self, strategies, metrics):
        def score_strategy(strategy, metrics):
            return sum(strategy[f'net_{metric}'] for metric in metrics)

        scored_strategies = [(strategy, score_strategy(strategy, metrics)) for strategy in strategies]
        return sorted(scored_strategies, key=lambda x: x[1], reverse=True)

    def filter_by_profit_loss_ratio(self, strategies, min_ratio):
        return [
            strategy for strategy in strategies
            if (strategy['max_profit'] != float('inf') and 
                strategy['max_loss'] != float('inf') and
                strategy['max_loss'] != 0 and
                (strategy['max_profit'] / strategy['max_loss']) >= min_ratio) or
               (strategy['max_loss'] == 0 and strategy['max_profit'] > 0)
        ]

    def analyze_strategies(self, max_legs=3, risk_profile='limited', min_profit_loss_ratio=0, metrics=['gamma', 'vega', 'theta']):
        all_strategies = self.create_strategies(max_legs)
        filtered_strategies = self.filter_strategies(all_strategies, risk_profile)
        ratio_filtered_strategies = self.filter_by_profit_loss_ratio(filtered_strategies, min_profit_loss_ratio)
        return self.optimize_strategies(ratio_filtered_strategies, metrics)

# def analyze_chunk(chunk, metrics, risk_profile):
#     analyzer = OptionsStrategyAnalyzer(chunk)
#     strategies = analyzer.create_strategies()
#     filtered_strategies = analyzer.filter_strategies(strategies, risk_profile)
#     optimized_strategies = analyzer.optimize_strategies(filtered_strategies, metrics)
#     return optimized_strategies


def analyze_chunk(chunk, max_legs, risk_profile, min_profit_loss_ratio, metrics):
    analyzer = OptionsStrategyAnalyzer(chunk)
    return analyzer.analyze_strategies(
        max_legs=max_legs,
        risk_profile=risk_profile,
        min_profit_loss_ratio=min_profit_loss_ratio,
        metrics=metrics
    )

def main():
    parser = argparse.ArgumentParser(description='Analyze options strategies')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('expiry_date', type=str, help='Option expiry date (YYYY-MM-DD)')
    parser.add_argument('--metrics', nargs='+', choices=['gamma', 'vega', 'theta'], default=['gamma'], 
                        help='Metrics to optimize for')
    parser.add_argument('--risk_profile', choices=['limited', 'unlimited', 'both'], default='both',
                        help='Risk profile of strategies to include')
    parser.add_argument('--profit_loss_ratio', type=float, default=2, 
                        help='Minimum ratio of max profit to max loss')
    parser.add_argument('--top', type=int, default=10, help='Number of top strategies to return')
    parser.add_argument('--range', choices=['ITM', 'NTM', 'OTM', 'SAK', 'SBK', 'SNK', 'ALL'], default='ALL', help='Option range')
    parser.add_argument('--max_legs', type=int, default=3, help='Maximum number of legs in a strategy')
    args = parser.parse_args()

    # Use the existing SchwabOptionsDataFetcher to get options data
    fetcher = SchwabOptionsDataFetcher()
    options_data = fetcher.get_options_chain(symbol=args.ticker, fromDate=args.expiry_date, toDate=args.expiry_date, ranges=args.range, contract_type='ALL')
    options_list = fetcher.extract_option_data(options_data).to_dict('records')

    # Split the list into chunks for parallel processing
    num_chunks = multiprocessing.cpu_count()
    chunk_size = len(options_list) // num_chunks
    chunks = [options_list[i:i + chunk_size] for i in range(0, len(options_list), chunk_size)]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        partial_analyze = partial(analyze_chunk, 
                                  max_legs=args.max_legs,
                                  risk_profile=args.risk_profile,
                                  min_profit_loss_ratio=args.profit_loss_ratio,
                                  metrics=args.metrics)
        results = list(executor.map(partial_analyze, chunks))

    # Combine and sort results
    all_strategies = sorted([item for sublist in results for item in sublist], key=lambda x: x[1], reverse=True)

    # Prepare data for tabulate
    table_data = []
    for rank, (strategy, score) in enumerate(all_strategies[:args.top], 1):
        legs_str = "; ".join([f"{'Long' if leg['direction'] == 1 else 'Short'} {leg['option'].type.capitalize()} {leg['option'].strike}" for leg in strategy['legs']])
        table_data.append([
            rank,
            strategy['strategy_type'],
            legs_str,
            f"${strategy['max_loss']:.2f}",
            f"${strategy['max_profit']:.2f}",
            f"{strategy['max_profit'] / strategy['max_loss']:.2f}",
            f"${strategy['credit_debit']:.2f} {'Credit' if strategy['credit_debit'] > 0 else 'Debit'}",
            *[f"{strategy[f'net_{metric}']:.6f}" for metric in args.metrics],
            f"{score:.6f}"
        ])

    # Define headers for the table
    headers = ['Rank', 'Strategy Type', 'Legs', 'Max Loss', 'Max Profit', 'P/L Ratio', 'Credit/Debit'] + [f'Net {metric.capitalize()}' for metric in args.metrics] + ['Score']

    # # Print the table
    # print(f"\nTop {args.top} {args.risk_profile} risk strategies with profit/loss ratio >= {args.profit_loss_ratio}, optimized for {', '.join(args.metrics)}:")
    # print(tabulate(table_data, headers=headers, tablefmt='simple'))

    # Create a directory for output files if it doesn't exist
    output_dir = 'strategy_analysis_results'
    os.makedirs(output_dir, exist_ok=True)

    # Generate a filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/strategies_{args.ticker}_{args.expiry_date}_{timestamp}.txt"

    # Write the table to the file
    with open(filename, 'w') as f:
        f.write(f"Top {args.top} {args.risk_profile} risk strategies with profit/loss ratio >= {args.profit_loss_ratio}, optimized for {', '.join(args.metrics)}:\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))

    print(f"Results have been written to {filename}")

if __name__ == "__main__":
    main()