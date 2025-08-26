import datetime
import os
import pandas as pd
import tushare as ts
import time

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Initialize Tushare
TUSHARE_TOKEN = os.getenv("TUSHARE_API_KEY")
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# Global cache instance
_cache = get_cache()


def _make_tushare_request(func, **kwargs):
    """
    Make a Tushare API request with rate limiting handling.
    
    Args:
        func: Tushare API function to call
        **kwargs: Parameters for the function
    
    Returns:
        DataFrame: The response data
    
    Raises:
        Exception: If the request fails
    """
    try:
        return func(**kwargs)
    except Exception as e:
        raise Exception(f"Tushare API error: {str(e)}")


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch A-share price data from Tushare."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date}_{end_date}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    # If not in cache, fetch from Tushare
    try:
        # For A-shares, ticker format is '000001.SZ' or '600000.SH'
        ts_code = f"{ticker.split('.')[0]}.{ticker.split('.')[1] if '.' in ticker else 'SH'}"
        
        # Get daily data
        df = _make_tushare_request(
            pro.daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            return []
            
        # Convert to Price objects
        prices = [
            Price(
                time=row["trade_date"],
                open=row["open"],
                close=row["close"],
                high=row["high"],
                low=row["low"],
                volume=row["vol"]
            )
            for _, row in df.iterrows()
        ]
        
        # Cache the results
        _cache.set_prices(cache_key, [p.model_dump() for p in prices])
        return prices
        
    except Exception as e:
        raise Exception(f"Error fetching Tushare price data: {str(e)}")


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch A-share financial metrics from Tushare."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    # If not in cache, fetch from Tushare
    try:
        ts_code = f"{ticker.split('.')[0]}.{ticker.split('.')[1] if '.' in ticker else 'SH'}"
        
        # Get income statement data
        income_df = _make_tushare_request(
            pro.income,
            ts_code=ts_code,
            period=end_date[:4] + "1231" if period == "annual" else end_date,
            fields="ts_code,end_date,revenue,total_profit,n_income"
        )
        
        # Get balance sheet data
        balance_df = _make_tushare_request(
            pro.balancesheet,
            ts_code=ts_code,
            period=end_date[:4] + "1231" if period == "annual" else end_date,
            fields="ts_code,end_date,total_assets,total_liab,total_hldr_eqy_exc_min_int"
        )
        
        if income_df.empty or balance_df.empty:
            return []
            
        # Merge data and create FinancialMetrics objects
        merged_df = pd.merge(income_df, balance_df, on=["ts_code", "end_date"])
        financial_metrics = [
            FinancialMetrics(
                ticker=ticker,
                report_period=row["end_date"],
                revenue=row["revenue"],
                net_income=row["n_income"],
                total_assets=row["total_assets"],
                total_liabilities=row["total_liab"],
                shareholders_equity=row["total_hldr_eqy_exc_min_int"],
                period=period
            )
            for _, row in merged_df.iterrows()
        ][:limit]
        
        # Cache the results
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
        return financial_metrics
        
    except Exception as e:
        raise Exception(f"Error fetching Tushare financial metrics: {str(e)}")


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch A-share line items from Tushare."""
    try:
        ts_code = f"{ticker.split('.')[0]}.{ticker.split('.')[1] if '.' in ticker else 'SH'}"
        
        # Map common line items to Tushare fields
        field_map = {
            # 利润表字段
            "revenue": ("income", "total_revenue"),
            "net_income": ("income", "n_income"),
            "operating_income": ("income", "operate_profit"),
            
            # 资产负债表字段
            
            "total_debt": ("balancesheet", "total_liab"),
            "shareholders_equity": ("balancesheet", "total_hldr_eqy_exc_min_int"),
            "goodwill_and_intangible_assets": ("balancesheet", "intan_assets"),
            "research_and_development": ("balancesheet", "r_and_d"),
            
            # 现金流量表字段
            "free_cash_flow": ("cashflow", "free_cashflow"),
            "capital_expenditure": ("cashflow", "c_pay_acq_const_fiolta"),
            "cash_and_equivalents": ("cashflow", "c_cash_equ_end_period"),
            
            # 财务指标字段
            "return_on_invested_capital": ("fina_indicator", "roic"),
            "operating_margin": ("fina_indicator", "op_of_gr"),
            "gross_margin": ("fina_indicator", "gross_margin"),

            "outstanding_shares": ("daily_basic", "float_share"),
        }
        
        # Get available fields
        fields = [item for item in line_items if item in field_map]
        
        if not fields:
            return []
            
        # Get financial data based on period
        if period == "annual":
            # 多接口数据获取
            dfs = []
            for api_name in set([v[0] for v in field_map.values()]):
                api_fields = [field_map[f][1] for f in fields if field_map[f][0] == api_name]
                api = getattr(pro, api_name)

                if api=='float_share':
                    df = _make_tushare_request(
                        api,
                        ts_code=ts_code,
                        trade_date=end_date.replace("-", ""),
                        # period=end_date[:4] + "0630",
                        fields=",".join([f for f in api_fields])
                    )
                else:
                    df = _make_tushare_request(
                        api,
                        ts_code=ts_code,
                        # period=end_date[:4] + "0630",
                        fields=",".join(["end_date"] + [f for f in api_fields])
                    )
                dfs.append(df)
            # 合并前移除重复的 end_date 列（保留第一个）
            for i in range(1, len(dfs)):
                if 'end_date' in dfs[i].columns:
                    dfs[i] = dfs[i].drop(columns=['end_date'])
            df = pd.concat(dfs, axis=1)
        else:
            # 多接口季度数据获取
            dfs = []
            for api_name in set([v[0] for v in field_map.values()]):
                api_fields = [f for f in fields if field_map[f][0] == api_name]
                api = getattr(pro, f"{api_name}_q" if api_name != 'fina_indicator' else 'fina_indicator_q')
                
                df = _make_tushare_request(
                    api,
                    ts_code=ts_code,
                    period=end_date,
                    fields=",".join(["end_date"] + [field_map[f][1] for f in api_fields])
                )
                dfs.append(df)
            
            df = pd.concat(dfs, axis=1,join='inner')
        
        if df.empty:
            return []
            
        # Convert to LineItem objects
        search_results = []
        for _, row in df.iterrows():
            # 为每行数据创建一个字典，包含所有必要字段
            item_data = {
                "ticker": ticker,
                "report_period": str(row["end_date"]),
                "period": period,
                "currency": "CNY"
            }
            # 将 line_items 中的字段及对应值添加到字典中
            for item in line_items:
                if item in field_map and field_map[item][1] in row:
                    item_data[item] = row[field_map[item][1]]
            # 创建 LineItem 对象并添加到结果列表
            search_results.append(LineItem(**item_data))
        
        return search_results[:limit]
        
    except Exception as e:
        raise Exception(f"Error fetching Tushare line items: {str(e)}")


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch A-share insider trades from Tushare."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    try:
        ts_code = f"{ticker.split('.')[0]}.{ticker.split('.')[1] if '.' in ticker else 'SH'}"
        
        # Get top holders data (Tushare's closest equivalent to insider trades)
        df = _make_tushare_request(
            pro.top10_holders,
            ts_code=ts_code,
            end_date=end_date.replace("-", ""),
            start_date=start_date.replace("-", "") if start_date else None,
            fields="ts_code,ann_date,holder_name,hold_amount,hold_ratio"
        )
        
        if df.empty:
            return []
            
        # Convert to InsiderTrade objects with required fields
        insider_trades = [
            InsiderTrade(
                ticker=ticker,
                issuer="",
                name=row["holder_name"],
                title="",
                is_board_director=False,
                transaction_date=row["ann_date"],
                transaction_shares=row["hold_amount"],
                transaction_price_per_share=0.0,
                transaction_value=0.0,
                shares_owned_before_transaction=0.0,
                shares_owned_after_transaction=row["hold_amount"],
                security_title="",
                filing_date=row["ann_date"]
            )
            for _, row in df.iterrows()
        ][:limit]
        
        # Cache the results
        _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in insider_trades])
        return insider_trades
        
    except Exception as e:
        raise Exception(f"Error fetching Tushare insider trades: {str(e)}")


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch A-share company news from Tushare."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    try:
        ts_code = f"{ticker.split('.')[0]}.{ticker.split('.')[1] if '.' in ticker else 'SH'}"
        
        # Get news data from Tushare (using major news as example)
        df = _make_tushare_request(
            pro.major_news,
            ts_code=ts_code,
            end_date=end_date.replace("-", ""),
            start_date=start_date.replace("-", "") if start_date else None,
            fields="ts_code,title,pub_time,src"
        )
        
        if df.empty:
            return []
            
        # Convert to CompanyNews objects with required fields
        company_news = [
            CompanyNews(
                ticker=ticker,
                title=row["title"],
                author=row["src"],
                source=row["src"],
                date=row["pub_time"],
                url=row["src"]
            )
            for _, row in df.iterrows()
        ][:limit]
        
        # Cache the results
        _cache.set_company_news(cache_key, [news.model_dump() for news in company_news])
        return company_news
        
    except Exception as e:
        raise Exception(f"Error fetching Tushare company news: {str(e)}")


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch A-share market cap from Tushare."""
    try:
        ts_code = f"{ticker.split('.')[0]}.{ticker.split('.')[1] if '.' in ticker else 'SH'}"
        
        # Get daily basic data which includes market cap
        df = _make_tushare_request(
            pro.daily_basic,
            ts_code=ts_code,
            trade_date=end_date.replace("-", ""),
            fields="trade_date,total_mv"
        )
        
        if df.empty:
            return None
                
        return float(df.iloc[0]["total_mv"])
        
    except Exception as e:
        print(f"Error fetching Tushare market cap: {str(e)}")
        return None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
