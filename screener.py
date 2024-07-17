import os
import json
import requests
import pandas as pd
import numpy as np
from numba import jit
import concurrent.futures
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

# Set up logging
log_filename = f"options_strategy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

    def get_options_chain(self, symbol, fromDate, toDate, ranges, contract_type="CALL",):
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

def calculate_strategy_metrics(leg_data, directions):
    directions = np.array(directions)
    net_premium = np.sum(leg_data[:, 0] * directions)
    net_delta = np.sum(leg_data[:, 1] * directions)
    net_gamma = np.sum(leg_data[:, 2] * directions)
    net_theta = np.sum(leg_data[:, 3] * directions)
    net_vega = np.sum(leg_data[:, 4] * directions)
    return net_premium, net_delta, net_gamma, net_theta, net_vega

def assess_risk(strategy, stock_price):
    net_premium, net_delta, net_gamma, net_theta, net_vega = strategy[:5]
    legs = strategy[5:]
    num_legs = len(legs) // 3  # Each leg has type, strike, and direction
    
    call_legs = []
    put_legs = []
    
    for i in range(num_legs):
        option_type = legs[i*3]
        strike = legs[i*3 + 1]
        direction = legs[i*3 + 2]
        if option_type == 0:  # call
            call_legs.append((strike, direction))
        else:  # put
            put_legs.append((strike, direction))
    
    call_legs.sort(key=lambda x: x[0])
    put_legs.sort(key=lambda x: x[0])
    
    max_loss = float('inf')
    max_profit = float('inf')
    is_limited_risk = False
    is_limited_profit = False

    logging.debug(f"Assessing strategy with {num_legs} legs: {call_legs} calls, {put_legs} puts")

    # Handle single-leg strategies
    if num_legs == 1:
        if call_legs:
            strike, direction = call_legs[0]
            if direction == 1:  # Long call
                max_loss = abs(net_premium)
                is_limited_risk = True
                logging.debug("Long call: Limited risk")
            else:  # Short call
                max_profit = abs(net_premium)
                is_limited_profit = True
                logging.debug("Short call: Limited profit")
        elif put_legs:
            strike, direction = put_legs[0]
            if direction == 1:  # Long put
                max_loss = abs(net_premium)
                is_limited_risk = True
                logging.debug("Long put: Limited risk")
            else:  # Short put
                max_profit = abs(net_premium)
                is_limited_profit = True
                logging.debug("Short put: Limited profit")

    # Handle multi-leg strategies
    elif num_legs == 2:
        if len(call_legs) == 2:  # Call spread
            low_strike, low_direction = call_legs[0]
            high_strike, high_direction = call_legs[1]
            width = high_strike - low_strike
            if low_direction == 1 and high_direction == -1:  # Bull Call Spread
                max_loss = abs(net_premium)
                max_profit = width + net_premium
                is_limited_risk = is_limited_profit = True
                logging.debug(f"Bull Call Spread: Limited risk and profit. Max loss: {max_loss}, Max profit: {max_profit}")
            elif low_direction == -1 and high_direction == 1:  # Bear Call Spread
                max_loss = width - net_premium
                max_profit = abs(net_premium)
                is_limited_risk = is_limited_profit = True
                logging.debug(f"Bear Call Spread: Limited risk and profit. Max loss: {max_loss}, Max profit: {max_profit}")
        elif len(put_legs) == 2:  # Put spread
            low_strike, low_direction = put_legs[0]
            high_strike, high_direction = put_legs[1]
            width = high_strike - low_strike
            if low_direction == -1 and high_direction == 1:  # Bull Put Spread
                max_loss = width - net_premium
                max_profit = abs(net_premium)
                is_limited_risk = is_limited_profit = True
                logging.debug(f"Bull Put Spread: Limited risk and profit. Max loss: {max_loss}, Max profit: {max_profit}")
            elif low_direction == 1 and high_direction == -1:  # Bear Put Spread
                max_loss = abs(net_premium)
                max_profit = width + net_premium
                is_limited_risk = is_limited_profit = True
                logging.debug(f"Bear Put Spread: Limited risk and profit. Max loss: {max_loss}, Max profit: {max_profit}")
        else:  # Straddle or Strangle
            is_limited_risk = True
            is_limited_profit = False
            logging.debug("Straddle or Strangle: Limited risk, unlimited profit")

    logging.debug(f"Final assessment: max_loss={max_loss}, max_profit={max_profit}, is_limited_risk={is_limited_risk}, is_limited_profit={is_limited_profit}")
    return max_loss, max_profit, is_limited_risk, is_limited_profit

# def process_chunk(chunk, options_array, stock_price):
#     strategies = []
#     for legs, directions in chunk:
#         leg_data = options_array[list(legs)]
#         net_premium, net_delta, net_gamma, net_theta, net_vega = calculate_strategy_metrics(leg_data[:, :5], directions)
        
#         # Convert 'call' to 0 and 'put' to 1
#         option_types = [0 if opt_type == 'call' else 1 for opt_type in leg_data[:, 5]]
        
#         strategy = np.array([net_premium, net_delta, net_gamma, net_theta, net_vega] + 
#                             option_types +  # type (0 for call, 1 for put)
#                             list(leg_data[:, 6]) +  # strike
#                             list(directions), dtype=np.float64)
        
#         max_loss, max_profit, is_limited_risk, is_limited_profit = assess_risk(strategy, stock_price)
        
#         strategies.append({
#             'legs': [{'option_type': 'call' if options_array[leg, 5] == 'call' else 'put',
#                       'strike': options_array[leg, 6],
#                       'direction': 'long' if direction == 1 else 'short'}
#                      for leg, direction in zip(legs, directions)],
#             'net_premium': net_premium,
#             'net_delta': net_delta,
#             'net_gamma': net_gamma,
#             'net_theta': net_theta,
#             'net_vega': net_vega,
#             'max_loss': max_loss,
#             'max_profit': max_profit,
#             'is_limited_risk': is_limited_risk,
#             'is_limited_profit': is_limited_profit,
#             'score': 0  # Will be calculated later
#         })
#     return strategies

def create_multi_leg_strategies(options_data, stock_price, max_legs=2, num_threads=None):
    logging.info("Starting to create multi-leg strategies")
    
    # Separate calls and puts
    calls = options_data[options_data['type'] == 'call']
    puts = options_data[options_data['type'] == 'put']
    
    all_strategies = []
    
    # Generate single-leg strategies
    for index, row in options_data.iterrows():
        all_strategies.append(([index], [1]))  # Long
        all_strategies.append(([index], [-1]))  # Short
    
    logging.info(f"Generated {len(all_strategies)} single-leg strategies")
    
    # Generate two-leg strategies (including spreads)
    if max_legs >= 2:
        # Call spreads
        for i, call1 in calls.iterrows():
            for j, call2 in calls.iterrows():
                if call1['strike'] < call2['strike']:
                    all_strategies.append(([i, j], [1, -1]))  # Bull Call Spread
                    all_strategies.append(([i, j], [-1, 1]))  # Bear Call Spread
        
        # Put spreads
        for i, put1 in puts.iterrows():
            for j, put2 in puts.iterrows():
                if put1['strike'] < put2['strike']:
                    all_strategies.append(([i, j], [-1, 1]))  # Bull Put Spread
                    all_strategies.append(([i, j], [1, -1]))  # Bear Put Spread
        
        # Straddles and Strangles
        for i, call in calls.iterrows():
            for j, put in puts.iterrows():
                all_strategies.append(([i, j], [1, 1]))  # Long Straddle/Strangle
                all_strategies.append(([i, j], [-1, -1]))  # Short Straddle/Strangle
    
    logging.info(f"Generated a total of {len(all_strategies)} strategies")
    
    # Convert options_data to numpy array for faster access
    options_array = options_data[['price', 'delta', 'gamma', 'theta', 'vega', 'type', 'strike']].values
    
    def process_chunk(chunk):
        strategies = []
        for legs, directions in chunk:
            leg_data = options_array[list(legs)]
            net_premium, net_delta, net_gamma, net_theta, net_vega = calculate_strategy_metrics(leg_data[:, :5], directions)
            
            option_types = [0 if opt_type == 'call' else 1 for opt_type in leg_data[:, 5]]
            
            strategy = np.array([net_premium, net_delta, net_gamma, net_theta, net_vega] + 
                                option_types +  # type (0 for call, 1 for put)
                                list(leg_data[:, 6]) +  # strike
                                list(directions), dtype=np.float64)
            
            logging.debug(f"Assessing strategy: {strategy}")
            max_loss, max_profit, is_limited_risk, is_limited_profit = assess_risk(strategy, stock_price)
            
            strategy_info = {
                'legs': [{'option_type': 'call' if options_array[leg, 5] == 'call' else 'put',
                          'strike': options_array[leg, 6],
                          'direction': 'long' if direction == 1 else 'short'}
                         for leg, direction in zip(legs, directions)],
                'net_premium': net_premium,
                'net_delta': net_delta,
                'net_gamma': net_gamma,
                'net_theta': net_theta,
                'net_vega': net_vega,
                'max_loss': max_loss,
                'max_profit': max_profit,
                'is_limited_risk': is_limited_risk,
                'is_limited_profit': is_limited_profit,
                'score': 0  # Will be calculated later
            }
            strategies.append(strategy_info)
            logging.debug(f"Strategy info: {strategy_info}")
        return strategies
    
    # Determine number of threads
    if num_threads is None:
        try:
            num_threads = multiprocessing.cpu_count()
        except NotImplementedError:
            num_threads = 4  # Default to 4 if cpu_count() is not implemented
    num_threads = min(32, num_threads + 4)  # Heuristic for optimal thread count
    
    # Split the work into chunks
    chunk_size = math.ceil(len(all_strategies) / num_threads)
    chunks = [all_strategies[i:i + chunk_size] for i in range(0, len(all_strategies), chunk_size)]
    
    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Flatten the results
    strategies = [strategy for chunk_result in results for strategy in chunk_result]
    
    logging.info(f"Processed {len(strategies)} valid strategies")
    return strategies

def optimize_strategies(strategies, metrics):
    if not metrics:
        return strategies

    for strategy in strategies:
        strategy['score'] = sum(abs(strategy[f'net_{metric}']) for metric in metrics) / len(metrics)
    
    return sorted(strategies, key=lambda x: x['score'], reverse=True)

def main():
    parser = argparse.ArgumentParser(description='Optimize options strategies')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('expiry', type=str, help='Option expiration date (YYYY-MM-DD)')
    parser.add_argument('metrics', nargs='+', choices=['gamma', 'delta', 'vega', 'theta'], help='Metrics to optimize for')
    parser.add_argument('--top', type=int, default=10, help='Number of top strategies to display')
    parser.add_argument('--include-unlimited-risk', action='store_true', help='Include unlimited risk strategies')
    parser.add_argument('--include-unlimited-profit', action='store_true', help='Include unlimited profit strategies')
    parser.add_argument('--range', choices=['ITM', 'NTM', 'OTM', 'SAK', 'SBK', 'SNK', 'ALL'], default='ALL', help='Option range')
    args = parser.parse_args()
    print(f"Args: {args}")

    try:
        # Parse expiry date
        expiry_date = datetime.datetime.strptime(args.expiry, '%Y-%m-%d').date()
        if expiry_date <= datetime.datetime.now().date():
            print("Error: Expiry date must be in the future.")
            return

        print(f"Analyzing options for {args.ticker} expiring on {args.expiry}, optimizing for {', '.join(args.metrics)}. This may take a moment...")

        fetcher = SchwabOptionsDataFetcher()
        
        if not fetcher.access_token:
            print("Failed to initialize access token. Exiting.")
            return

        options_chain = fetcher.get_options_chain(symbol=args.ticker, fromDate=args.expiry, toDate=args.expiry, ranges=args.range)
        
        # Debug: Print the keys in the options chain
        print("Options chain keys:", options_chain.keys())
        
        options_data = fetcher.extract_option_data(options_chain)
        
        # Debug: Print the first few rows of extracted data
        print("Sample of extracted option data:")
        print(options_data.head())
        
        if options_data.empty:
            print("No options data found. Please check the ticker symbol and expiry date.")
            return
        
        stock_price = fetcher.get_stock_price(options_chain)
        print(f"Underlying stock price: {stock_price}")
        
        log_print(f"Underlying stock price: {stock_price}")
        all_strategies = create_multi_leg_strategies(options_data=options_data, stock_price=stock_price)
        log_print(f"Total strategies created: {len(all_strategies)}")

        limited_risk_strategies = [s for s in all_strategies if s['is_limited_risk']]
        limited_profit_strategies = [s for s in all_strategies if s['is_limited_profit']]
        limited_risk_and_profit_strategies = [s for s in all_strategies if s['is_limited_risk'] and s['is_limited_profit']]

        log_print(f"Strategies with limited risk: {len(limited_risk_strategies)}")
        log_print(f"Strategies with limited profit: {len(limited_profit_strategies)}")
        log_print(f"Strategies with both limited risk and limited profit: {len(limited_risk_and_profit_strategies)}")

        # Default to limited risk and profit strategies
        filtered_strategies = limited_risk_and_profit_strategies

        if args.include_unlimited_risk:
            filtered_strategies = limited_profit_strategies
            log_print("Including unlimited risk strategies.")

        if args.include_unlimited_profit:
            filtered_strategies = [s for s in filtered_strategies if s['is_limited_risk']]
            log_print("Including unlimited profit strategies.")

        if args.include_unlimited_risk and args.include_unlimited_profit:
            filtered_strategies = all_strategies
            log_print("Including all strategies (unlimited risk and unlimited profit).")

        log_print(f"Strategies meeting all criteria: {len(filtered_strategies)}")

        if not filtered_strategies:
            log_print("No suitable strategies found for the given parameters.")
            return

        optimized_strategies = optimize_strategies(filtered_strategies, args.metrics)

         # Prepare the results
        results = []
        for i, strategy in enumerate(optimized_strategies[:args.top], 1):
            risk_type = "Limited" if strategy['is_limited_risk'] else "Unlimited"
            profit_type = "Limited" if strategy['is_limited_profit'] else "Unlimited"
            row = [i, f"{len(strategy['legs'])}-leg {risk_type} Risk, {profit_type} Profit"]
            for metric in args.metrics:
                row.append(f"{strategy[f'net_{metric}']:.4f}")
            row.extend([
                # f"${strategy['max_loss']:.2f}" if strategy['max_loss'] != float('inf') else "Unlimited",
                # f"${strategy['max_profit']:.2f}" if strategy['max_profit'] != float('inf') else "Unlimited",
                f"${strategy['max_loss']:.2f}",
                f"${strategy['max_profit']:.2f}",
                f"{strategy['score']:.4f}"
            ])
            results.append(row)

        # Pretty print the results
        headers = ['Rank', 'Strategy'] + [m.capitalize() for m in args.metrics] + ['Max Loss', 'Max Profit', 'Score']
        print(tabulate(results, headers=headers, tablefmt='grid'))

        # Print detailed information for top strategy
        top_strategy = optimized_strategies[0]
        print("\nTop Strategy Details:")
        for i, leg in enumerate(top_strategy['legs'], 1):
            print(f"Leg {i}: {leg['direction'].capitalize()} {leg['option_type']} @ strike {leg['strike']}")

    except ValueError as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()