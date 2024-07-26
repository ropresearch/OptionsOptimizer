import os
import requests
import pandas as pd
import argparse
from tabulate import tabulate
from dotenv import load_dotenv
from auth import get_access_token
import logging
from datetime import datetime
import itertools

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OptionsDataFetcher:
    def __init__(self):
        load_dotenv()
        self.MARKET_BASE_URL = os.getenv('MARKET_BASE_URL')
        self.access_token = get_access_token()

    def get_options_chain(self, symbol, expiry_date, ranges="ALL"):
        if not self.MARKET_BASE_URL or not self.access_token:
            raise ValueError("Missing MARKET_BASE_URL or access token")
        
        params = {
            'symbol': symbol,
            'fromDate': expiry_date,
            'toDate': expiry_date,
            'contractType': 'ALL',
            'includeUnderlyingQuote': 'true',
            'strategy': 'ANALYTICAL',
            'range': ranges
        }
        
        response = requests.get(f'{self.MARKET_BASE_URL}/chains', 
                                headers={'Authorization': f'Bearer {self.access_token}'}, 
                                params=params)
        response.raise_for_status()
        return response.json()

    def extract_option_data(self, json_data):
        options_data = []
        underlying_price = json_data.get('underlyingPrice', 0)
        for exp_map_key in ['callExpDateMap', 'putExpDateMap']:
            if exp_map_key in json_data:
                for date, strikes in json_data[exp_map_key].items():
                    for strike, options in strikes.items():
                        for option in options:
                            option_type = 'call' if exp_map_key == 'callExpDateMap' else 'put'
                            strike_price = float(strike)
                            option_price = option.get('last', 0)
                            
                            # Calculate max profit and loss
                            if option_type == 'call':
                                max_profit = float('inf')
                                max_loss = option_price
                            else:  # put
                                max_profit = strike_price - option_price
                                max_loss = option_price

                            options_data.append({
                                'symbol': option.get('symbol'),
                                'type': option_type,
                                'strike': strike_price,
                                'price': option_price,
                                'delta': option.get('delta', 0),
                                'gamma': option.get('gamma', 0),
                                'theta': option.get('theta', 0),
                                'vega': option.get('vega', 0),
                                'rho': option.get('rho', 0),  # Added rho
                                'impliedVolatility': option.get('volatility', 0),
                                'max_profit': max_profit,
                                'max_loss': max_loss,
                            })
        return pd.DataFrame(options_data)

class OptionsScreener:
    def __init__(self, options_data, min_option_price=0.05):
        self.options = options_data[options_data['price'] >= min_option_price]

    def screen_single_leg_options(self, metrics, min_profit_loss_ratio=1):
        self.options['profit_loss_ratio'] = self.options.apply(
            lambda row: row['max_profit'] / row['max_loss'] if row['max_loss'] != 0 else float('inf'), 
            axis=1
        )
        filtered_options = self.options[self.options['profit_loss_ratio'] >= min_profit_loss_ratio]
        
        filtered_options['score'] = filtered_options[metrics].abs().sum(axis=1)
        
        return filtered_options.sort_values('score', ascending=False)

    def generate_debit_spreads(self):
        calls = self.options[self.options['type'] == 'call'].sort_values('strike')
        puts = self.options[self.options['type'] == 'put'].sort_values('strike')

        spreads = []

        # Bull Call Spreads
        for (_, long), (_, short) in itertools.combinations(calls.iterrows(), 2):
            if long['strike'] < short['strike']:
                spread = self._create_spread(long, short, 'bull_call')
                if spread is not None:
                    spreads.append(spread)

        # Bear Put Spreads
        for (_, long), (_, short) in itertools.combinations(puts.iterrows(), 2):
            if long['strike'] > short['strike']:
                spread = self._create_spread(long, short, 'bear_put')
                if spread is not None:
                    spreads.append(spread)

        return pd.DataFrame(spreads)

    def _create_spread(self, long, short, spread_type):
        spread = {
            'strategy': spread_type,
            'long_symbol': long['symbol'],
            'short_symbol': short['symbol'],
            'long_strike': long['strike'],
            'short_strike': short['strike'],
            'net_debit': long['price'] - short['price'],
        }

        if spread_type == 'bull_call':
            spread['max_profit'] = short['strike'] - long['strike'] - spread['net_debit']
            spread['max_loss'] = spread['net_debit']
        else:  # bear_put
            spread['max_profit'] = long['strike'] - short['strike'] - spread['net_debit']
            spread['max_loss'] = spread['net_debit']

        if spread['max_loss'] <= 0:
            return None  # Skip this spread as it's not valid

        spread['profit_loss_ratio'] = spread['max_profit'] / spread['max_loss']

        # Calculate net Greeks
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:  # Added rho
            spread[f'net_{greek}'] = long[greek] - short[greek]

        return spread

    def screen_debit_spreads(self, metrics, min_profit_loss_ratio=1):
        spreads = self.generate_debit_spreads()
        if spreads.empty:
            return pd.DataFrame()  # Return an empty DataFrame if no valid spreads are found
        filtered_spreads = spreads[spreads['profit_loss_ratio'] >= min_profit_loss_ratio]
        
        if filtered_spreads.empty:
            return pd.DataFrame()  # Return an empty DataFrame if no spreads meet the criteria
        filtered_spreads['score'] = filtered_spreads[[f'net_{metric}' for metric in metrics]].abs().sum(axis=1)
        
        return filtered_spreads.sort_values('score', ascending=False)

def main():
    parser = argparse.ArgumentParser(description='Screen for options strategies')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('expiry_date', type=str, help='Option expiry date (YYYY-MM-DD)')
    parser.add_argument('--max_legs', type=int, choices=[1, 2], default=2, 
                        help='Maximum number of legs for strategies to screen (1 for single-leg, 2 for single-leg and spreads)')
    parser.add_argument('--metrics', nargs='+', choices=['delta', 'gamma', 'vega', 'theta', 'rho'],  # Added rho
                        default=['gamma', 'vega', 'theta'], help='Metrics to optimize for')
    parser.add_argument('--range', choices=['ITM', 'NTM', 'OTM', 'SAK', 'SBK', 'SNK', 'ALL'], 
                        default='ALL', help='Option range')
    parser.add_argument('--top', type=int, default=10, help='Number of top strategies to return')
    parser.add_argument('--min_option_price', type=float, default=0.05, 
                        help='Minimum price for options to consider')
    parser.add_argument('--min_profit_loss_ratio', type=float, default=1, 
                        help='Minimum profit/loss ratio')
    args = parser.parse_args()

    fetcher = OptionsDataFetcher()
    options_data = fetcher.get_options_chain(args.ticker, args.expiry_date, args.range)
    options_df = fetcher.extract_option_data(options_data)

    screener = OptionsScreener(options_df, args.min_option_price)

    results = []

    if args.max_legs >= 1:
        single_leg_options = screener.screen_single_leg_options(args.metrics, args.min_profit_loss_ratio)
        results.append(('Single-Leg Options', single_leg_options))

    if args.max_legs == 2:
        debit_spreads = screener.screen_debit_spreads(args.metrics, args.min_profit_loss_ratio)
        results.append(('Debit Spreads', debit_spreads))

    # Save results to file
    output_dir = 'strategy_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/options_{args.ticker}_{args.expiry_date}_{timestamp}.txt"

    with open(filename, 'w') as f:
        for strategy_name, data in results:
            f.write(f"\nTop {args.top} {strategy_name} for {args.ticker}, expiry {args.expiry_date}, "
                    f"optimized for {', '.join(args.metrics)}, min P/L ratio {args.min_profit_loss_ratio}:\n\n")
            
            if strategy_name == 'Single-Leg Options':
                headers = ['Symbol', 'Type', 'Strike', 'Price', 'P/L Ratio'] + [metric.capitalize() for metric in args.metrics] + ['Score']
                table_data = data[['symbol', 'type', 'strike', 'price', 'profit_loss_ratio'] + args.metrics + ['score']].head(args.top).values.tolist()
            else:  # Debit Spreads
                headers = ['Strategy', 'Long Symbol', 'Short Symbol', 'Long Strike', 'Short Strike', 'Net Debit', 'Max Profit', 'Max Loss', 'P/L Ratio'] + [f'Net {metric.capitalize()}' for metric in args.metrics] + ['Score']
                table_data = data[['strategy', 'long_symbol', 'short_symbol', 'long_strike', 'short_strike', 'net_debit', 'max_profit', 'max_loss', 'profit_loss_ratio'] + [f'net_{metric}' for metric in args.metrics] + ['score']].head(args.top).values.tolist()

            f.write(tabulate(table_data, headers=headers, floatfmt=".4f", tablefmt="grid"))
            f.write('\n\n')

            # Also print to console
            print(f"\nTop {args.top} {strategy_name}:")
            print(tabulate(table_data, headers=headers, floatfmt=".4f", tablefmt="grid"))

    print(f"\nResults have been written to {filename}")

if __name__ == "__main__":
    main()