import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
from io import StringIO

# Import the new screener components
from screener3 import OptionsDataFetcher, OptionsScreener
import traceback

load_dotenv()

bot = commands.Bot(command_prefix='/', intents=discord.Intents.all())

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.command(name='analyze')
async def analyze_options(ctx, ticker: str, expiry_date: str, *args):
    await ctx.send(f"Analyzing options for {ticker}. This may take a few moments...")

    # Default values
    metrics = ['gamma', 'vega', 'theta']
    range_value = 'ALL'
    top = 10
    min_option_price = 0.05
    min_profit_loss_ratio = 1.0
    max_legs = 2

    # Parse additional arguments
    try:
        for arg in args:
            if arg.lower().startswith(('gamma', 'vega', 'theta', 'delta', 'rho')):
                metrics = [m.strip() for m in arg.lower().split(',')]
            elif arg.upper() in ['ITM', 'NTM', 'OTM', 'SAK', 'SBK', 'SNK', 'ALL']:
                range_value = arg.upper()
            elif arg.startswith('top='):
                top = int(arg.split('=')[1])
            elif arg.startswith('min_price='):
                min_option_price = float(arg.split('=')[1])
            elif arg.startswith('min_pl_ratio='):
                min_profit_loss_ratio = float(arg.split('=')[1])
            elif arg.startswith('max_legs='):
                max_legs = int(arg.split('=')[1])
    except ValueError:
        await ctx.send("Error: Invalid argument format. Please check your command and try again.")
        return

    try:
        fetcher = OptionsDataFetcher()
        options_data = fetcher.get_options_chain(symbol=ticker, expiry_date=expiry_date, ranges=range_value)
        options_df = fetcher.extract_option_data(options_data)

        if options_df.empty:
            await ctx.send(f"No options data found for {ticker} expiring on {expiry_date}. Please check the ticker and expiry date.")
            return

        screener = OptionsScreener(options_df, min_option_price=min_option_price)

        results = []
        if max_legs >= 1:
            single_leg_options = screener.screen_single_leg_options(metrics, min_profit_loss_ratio).head(top)
            results.append(('Single-Leg Options', single_leg_options))
        if max_legs == 2:
            debit_spreads = screener.screen_debit_spreads(metrics, min_profit_loss_ratio).head(top)
            results.append(('Debit Spreads', debit_spreads))

        if all(data.empty for _, data in results):
            await ctx.send(f"No options meeting the criteria found for {ticker}. Try adjusting your parameters.")
            return

        for strategy_name, data in results:
            embeds = []
            current_embed = discord.Embed(title=f"Top {strategy_name} for {ticker} ({expiry_date})", 
                                          description=f"Metrics: {', '.join(metrics)}, Min P/L Ratio: {min_profit_loss_ratio}")

            for rank, (_, option) in enumerate(data.iterrows(), 1):
                if strategy_name == 'Single-Leg Options':
                    strategy_field = f'''
                        Type: {option['type'].capitalize()}
                        Strike: ${option['strike']:.2f}
                        Price: ${option['price']:.2f}
                        P/L Ratio: {option['profit_loss_ratio']:.2f}
                        Max Profit: ${option['max_profit']:.2f}
                        Max Loss: ${option['max_loss']:.2f}
                        {', '.join([f"{metric.capitalize()}: {option[metric]:.6f}" for metric in metrics])}
                        Score: {option['score']:.6f}
                    '''
                    field_name = f"#{rank} - {option['symbol']}"
                else:  # Debit Spreads
                    strategy_field = f'''
                        Strategy: {option['strategy']}
                        Long Strike: ${option['long_strike']:.2f}
                        Short Strike: ${option['short_strike']:.2f}
                        Net Debit: ${option['net_debit']:.2f}
                        P/L Ratio: {option['profit_loss_ratio']:.2f}
                        Max Profit: ${option['max_profit']:.2f}
                        Max Loss: ${option['max_loss']:.2f}
                        {', '.join([f"Net {metric.capitalize()}: {option[f'net_{metric}']:.6f}" for metric in metrics])}
                        Score: {option['score']:.6f}
                    '''
                    field_name = f"#{rank} - {option['long_symbol']}/{option['short_symbol']}"

                if len(current_embed) + len(strategy_field) > 6000:
                    embeds.append(current_embed)
                    current_embed = discord.Embed(title=f"Top {strategy_name} for {ticker} ({expiry_date}) Continued", 
                                                  description=f"Metrics: {', '.join(metrics)}, Min P/L Ratio: {min_profit_loss_ratio}")

                current_embed.add_field(name=field_name, value=strategy_field, inline=False)

            embeds.append(current_embed)

            for embed in embeds:
                await ctx.send(embed=embed)

    except Exception as e:
        error_message = f"An error occurred while analyzing options for {ticker}: {str(e)}"
        await ctx.send(error_message)
        print(f"Detailed error: {traceback.format_exc()}")  # Log the full error for debugging

bot.run(os.getenv('DISCORD_TOKEN'))