import csv, random, itertools
random.seed(7)
S=[]; seen=set()
def put(cat,s):
    s=s.strip(); s=s[0].upper()+s[1:]
    if not s.endswith("."): s+="."
    if s not in seen: seen.add(s); S.append((cat,s))
def art(w): return "an" if w[0].lower() in "aeiou" else "a"

# ---------- instruments ----------
# name, category, issuer/market, risk, income, liquidity, traits
INS=[
("common stock","equity security","issued by corporations","relatively high risk","dividends and capital gains","usually high liquidity when listed on an exchange",
 ["it represents a residual ownership claim on a company's assets and earnings","its holders usually have voting rights at shareholder meetings","in liquidation its holders are paid only after creditors and preferred shareholders"]),
("preferred stock","hybrid security","issued by corporations","moderate risk","fixed or floating dividends","variable liquidity",
 ["its dividends must usually be paid before any dividends on common stock","it often carries no voting rights","cumulative preferred shares accumulate unpaid dividends as arrears"]),
("government bond","debt security","issued by national governments","low credit risk when issued by a stable sovereign in its own currency","periodic coupon payments","high liquidity in major markets",
 ["its price moves inversely to changes in market interest rates","its yield is often used as a benchmark for the risk-free rate","it is still exposed to interest rate risk and inflation risk"]),
("corporate bond","debt security","issued by companies","moderate risk that depends on the issuer's credit rating","periodic coupon payments","moderate liquidity",
 ["its yield includes a credit spread over comparable government bonds","bonds rated below investment grade are called high-yield bonds","its holders rank ahead of shareholders in a bankruptcy"]),
("zero-coupon bond","debt security","issued by governments and companies","interest rate risk that rises with maturity","the difference between the purchase price and face value","variable liquidity",
 ["it pays no periodic coupons and is sold at a discount to its face value","its Macaulay duration equals its time to maturity","its price is especially sensitive to changes in interest rates"]),
("treasury bill","money market instrument","issued by national governments","very low risk","the discount between the purchase price and face value","very high liquidity",
 ["it has a maturity of one year or less","it is sold at a discount and repaid at par","it is commonly used as a proxy for a short-term risk-free asset"]),
("certificate of deposit","time deposit","offered by banks","low risk","fixed interest","low liquidity before maturity",
 ["it requires the depositor to leave funds in place for a fixed term","early withdrawal usually triggers a penalty","it may be covered by a deposit guarantee scheme up to a legal limit"]),
("mutual fund","pooled investment vehicle","managed by asset management companies","risk that depends on its underlying holdings","distributions and changes in net asset value","daily redemption at net asset value in most open-ended funds",
 ["it pools money from many investors to buy a diversified portfolio","its units are priced at net asset value calculated at the end of each trading day","it charges management fees summarized by its expense ratio"]),
("exchange-traded fund","pooled investment vehicle","listed on stock exchanges","risk that depends on its underlying index or holdings","distributions and price changes","intraday liquidity on an exchange",
 ["it trades on an exchange throughout the day like a stock","many are designed to track a market index passively","an arbitrage mechanism involving authorized participants keeps its price close to its net asset value"]),
("futures contract","derivative","traded on organized exchanges","high risk due to leverage","gains or losses from price changes in the underlying asset","high liquidity for standardized contracts",
 ["it obliges both parties to buy or sell an asset at a set price on a future date","it is marked to market daily, with gains and losses settled through margin accounts","a clearinghouse acts as the counterparty to both sides, reducing counterparty risk"]),
("forward contract","derivative","traded over the counter","counterparty risk and market risk","gains or losses at settlement","low liquidity",
 ["it is a customized agreement to exchange an asset at a set price on a future date","unlike a futures contract, it is usually not marked to market daily","companies often use it to hedge foreign exchange exposure"]),
("call option","derivative","traded on exchanges and over the counter","high risk for the buyer, whose maximum loss is the premium","payoff when the underlying price rises above the strike price","variable liquidity",
 ["it gives its holder the right, but not the obligation, to buy the underlying asset at the strike price","its buyer pays a premium to the seller, called the writer","its value increases with the volatility of the underlying asset"]),
("put option","derivative","traded on exchanges and over the counter","high risk for the buyer, whose maximum loss is the premium","payoff when the underlying price falls below the strike price","variable liquidity",
 ["it gives its holder the right, but not the obligation, to sell the underlying asset at the strike price","it is commonly used to protect a portfolio against falling prices","a protective put combines a long stock position with a long put"]),
("interest rate swap","derivative","traded over the counter","counterparty risk and interest rate risk","net exchanged interest payments","moderate liquidity for standard maturities",
 ["it exchanges fixed-rate interest payments for floating-rate payments on a notional principal","the notional principal itself is usually not exchanged","firms use it to transform floating-rate debt into fixed-rate debt"]),
("real estate investment trust","pooled investment vehicle","that owns or finances income-producing property","moderate to high risk","rental income distributions and price changes","high liquidity when listed",
 ["it is often required to distribute a large share of its taxable income to shareholders","it allows investors to gain exposure to property without buying buildings directly","its value is sensitive to interest rates and property markets"]),
("mortgage-backed security","asset-backed security","created by securitizing pools of home loans","credit risk and prepayment risk","principal and interest payments passed through from borrowers","variable liquidity",
 ["its cash flows depend on the payments made by the underlying mortgage borrowers","falling interest rates tend to increase prepayments","it played a central role in the global financial crisis of 2007 to 2009"]),
("convertible bond","hybrid security","issued by companies","moderate risk","coupon payments plus the option to convert into shares","moderate liquidity",
 ["it can be exchanged for a predetermined number of the issuer's shares","it usually pays a lower coupon than a comparable straight bond","its value combines a straight bond value and an embedded call option"]),
("money market fund","pooled investment vehicle","managed by asset management companies","low risk","short-term interest income","high liquidity",
 ["it invests in short-term, high-quality debt instruments","it aims to preserve capital while providing liquidity","it is not the same as an insured bank deposit"]),
]
for n,cat,iss,risk,inc,liq,tr in INS:
    put("instruments",f"{art(n).capitalize()} {n} is {art(cat)} {cat}")
    put("instruments",f"{art(n).capitalize()} {n} is typically {iss}")
    put("instruments",f"The {n} is classified as {art(cat)} {cat}")
    put("instruments",f"Investors in {art(n)} {n} face {risk}")
    put("instruments",f"In terms of risk, {art(n)} {n} carries {risk}")
    put("instruments",f"The return on {art(n)} {n} comes from {inc}")
    put("instruments",f"{art(n).capitalize()} {n} generates income through {inc}")
    put("instruments",f"Regarding liquidity, {art(n)} {n} offers {liq}")
    for t in tr:
        put("instruments",f"A key feature of {art(n)} {n} is that {t}")
        put("instruments",f"Regarding the {n}, {t}")
for (a,*x),(b,*y) in itertools.permutations(INS,2):
    if x[0]!=y[0]:
        put("comparative",f"{art(a).capitalize()} {a} is {art(x[0])} {x[0]}, whereas {art(b)} {b} is {art(y[0])} {y[0]}")
    put("comparative",f"The return on {art(a)} {a} comes from {x[3]}, while the return on {art(b)} {b} comes from {y[3]}")

# ---------- ratios ----------
RAT=[
("price-to-earnings ratio","market price per share divided by earnings per share","valuation","how much investors pay for each unit of earnings"),
("price-to-book ratio","market price per share divided by book value per share","valuation","the market value of equity relative to its accounting value"),
("dividend yield","annual dividends per share divided by the share price","valuation","the cash income return from holding a share"),
("earnings per share","net income available to common shareholders divided by the weighted average number of common shares outstanding","profitability","the profit attributable to each common share"),
("return on equity","net income divided by average shareholders' equity","profitability","how efficiently a company generates profit from shareholders' capital"),
("return on assets","net income divided by average total assets","profitability","how efficiently a company uses its assets to generate profit"),
("gross margin","gross profit divided by revenue","profitability","the share of revenue remaining after the cost of goods sold"),
("operating margin","operating income divided by revenue","profitability","the share of revenue remaining after operating expenses"),
("net profit margin","net income divided by revenue","profitability","the share of revenue that becomes net profit"),
("current ratio","current assets divided by current liabilities","liquidity","a company's ability to meet short-term obligations"),
("quick ratio","current assets minus inventories, divided by current liabilities","liquidity","short-term liquidity without relying on the sale of inventory"),
("cash ratio","cash and cash equivalents divided by current liabilities","liquidity","the ability to pay short-term liabilities using only cash"),
("debt-to-equity ratio","total debt divided by shareholders' equity","leverage","the degree to which a company finances itself with debt relative to equity"),
("interest coverage ratio","earnings before interest and taxes divided by interest expense","solvency","how easily a company can pay interest on its debt"),
("asset turnover","revenue divided by average total assets","efficiency","how effectively a company uses its assets to generate sales"),
("inventory turnover","cost of goods sold divided by average inventory","efficiency","how many times inventory is sold and replaced during a period"),
("days sales outstanding","accounts receivable divided by revenue, multiplied by the number of days in the period","efficiency","the average number of days needed to collect payment from customers"),
("enterprise value to EBITDA","enterprise value divided by earnings before interest, taxes, depreciation, and amortization","valuation","a company's total value relative to its operating cash earnings"),
("payout ratio","dividends divided by net income","dividend policy","the share of earnings paid out to shareholders as dividends"),
("net debt to EBITDA","total debt minus cash, divided by EBITDA","leverage","roughly how many years of operating earnings would be needed to repay net debt"),
]
for n,f,g,m in RAT:
    put("ratios",f"The {n} is calculated as {f}")
    put("ratios",f"To compute the {n}, divide as follows: {f}")
    put("ratios",f"The {n} is {art(g)} {g} ratio")
    put("ratios",f"The {n} measures {m}")
    put("ratios",f"Analysts use the {n} to assess {m}")
    put("ratios",f"In financial statement analysis, the {n}, defined as {f}, indicates {m}")
    put("ratios",f"The {n} belongs to the family of {g} metrics")

# ---------- concepts ----------
concepts="""The time value of money states that a sum of money today is worth more than the same sum in the future.
Present value is the current worth of a future cash flow discounted at an appropriate rate.
Future value is the amount an investment will grow to after earning interest over a given period.
Compound interest is interest calculated on both the initial principal and the accumulated interest.
Simple interest is calculated only on the original principal.
The effective annual rate accounts for the effect of compounding within a year.
Net present value is the sum of the present values of all cash inflows and outflows of a project.
A project with a positive net present value is expected to add value to the firm.
The internal rate of return is the discount rate at which a project's net present value equals zero.
The payback period is the time required for cumulative cash flows to recover the initial investment.
The weighted average cost of capital is the average rate a company pays to finance its assets, weighted by the proportion of debt and equity.
The capital asset pricing model relates an asset's expected return to its systematic risk, measured by beta.
Beta measures the sensitivity of an asset's returns to movements in the overall market.
An asset with a beta greater than one tends to move more than the market.
Systematic risk affects the entire market and cannot be eliminated by diversification.
Unsystematic risk is specific to a company or industry and can be reduced through diversification.
Diversification reduces portfolio risk by combining assets whose returns are not perfectly correlated.
The efficient frontier represents portfolios that offer the highest expected return for each level of risk.
Modern portfolio theory was developed by Harry Markowitz in the 1950s.
The Sharpe ratio measures excess return per unit of total risk.
Standard deviation of returns is a common measure of investment volatility.
Correlation measures how closely the returns of two assets move together.
The efficient market hypothesis states that asset prices reflect all available information.
Arbitrage is the simultaneous purchase and sale of an asset to profit from a price difference.
Liquidity risk is the risk that an asset cannot be sold quickly without a significant price concession.
Credit risk is the risk that a borrower fails to make required payments.
Interest rate risk is the risk that changes in interest rates reduce the value of an investment.
Inflation risk is the risk that rising prices erode the purchasing power of returns.
Currency risk arises when the value of an investment changes because of exchange rate movements.
Counterparty risk is the risk that the other party in a transaction defaults on its obligations.
Leverage amplifies both gains and losses by using borrowed funds.
Duration measures the sensitivity of a bond's price to changes in interest rates.
Convexity describes how a bond's duration changes as interest rates change.
Bond prices fall when market interest rates rise.
A bond trading above its face value is said to trade at a premium.
A bond trading below its face value is said to trade at a discount.
The yield to maturity is the discount rate that equates a bond's price with the present value of its cash flows.
The coupon rate is the annual interest payment expressed as a percentage of a bond's face value.
The yield curve plots the yields of bonds with equal credit quality across different maturities.
An inverted yield curve occurs when short-term yields exceed long-term yields.
Credit rating agencies assess the creditworthiness of bond issuers.
The balance sheet reports a company's assets, liabilities, and equity at a specific point in time.
The income statement reports a company's revenues, expenses, and profit over a period.
The cash flow statement shows cash flows from operating, investing, and financing activities.
The accounting equation states that assets equal liabilities plus shareholders' equity.
Accrual accounting records revenues when earned and expenses when incurred, regardless of cash movements.
Depreciation allocates the cost of a tangible asset over its useful life.
Amortization allocates the cost of an intangible asset over its useful life.
Working capital equals current assets minus current liabilities.
Free cash flow is the cash a company generates after capital expenditures.
Retained earnings are accumulated profits that have not been distributed as dividends.
Goodwill arises when a company acquires another business for more than the fair value of its identifiable net assets.
EBITDA stands for earnings before interest, taxes, depreciation, and amortization.
International Financial Reporting Standards are accounting standards used in many countries, including the European Union.
Generally accepted accounting principles in the United States are known as US GAAP.
A central bank manages a country's monetary policy and money supply.
The European Central Bank sets monetary policy for the euro area.
Raising policy interest rates is a common tool used by central banks to reduce inflation.
Quantitative easing involves a central bank buying financial assets to increase the money supply.
Inflation is the general increase in prices and the fall in the purchasing power of money.
Deflation is a sustained decrease in the general price level.
Gross domestic product measures the total value of goods and services produced in an economy.
The consumer price index tracks changes in the price of a basket of consumer goods and services.
Fiscal policy refers to government decisions on taxation and public spending.
A recession is commonly described as a significant decline in economic activity lasting several months.
The nominal interest rate is the stated rate before adjusting for inflation.
The real interest rate is the nominal interest rate adjusted for inflation.
The Fisher equation links nominal interest rates, real interest rates, and expected inflation.
An exchange rate is the price of one currency expressed in terms of another.
Commercial banks accept deposits and make loans.
Fractional reserve banking allows banks to lend out a portion of their deposits.
Capital adequacy requirements oblige banks to hold capital in proportion to their risk-weighted assets.
The Basel Accords are international standards for bank capital and liquidity regulation.
Anti-money laundering rules require financial institutions to detect and report suspicious transactions.
Know your customer procedures require financial institutions to verify the identity of their clients.
Customer due diligence involves assessing the risks associated with a business relationship.
A primary market is where securities are issued and sold for the first time.
A secondary market is where previously issued securities are traded between investors.
An initial public offering is the first sale of a company's shares to the public.
A market order is executed immediately at the best available price.
A limit order is executed only at a specified price or better.
The bid-ask spread is the difference between the highest price a buyer will pay and the lowest price a seller will accept.
Market makers provide liquidity by quoting both bid and ask prices.
Short selling involves selling borrowed securities in the expectation of buying them back at a lower price.
A stock index tracks the performance of a selected group of shares.
Market capitalization equals the share price multiplied by the number of shares outstanding.
A bull market is a period of generally rising prices.
A bear market is often defined as a decline of at least 20 percent from a recent peak.
A stock split increases the number of shares outstanding without changing the company's market value.
A share buyback reduces the number of shares outstanding.
Dollar-cost averaging involves investing a fixed amount at regular intervals.
Asset allocation is the division of a portfolio among asset classes such as equities, bonds, and cash.
Rebalancing restores a portfolio to its target asset allocation.
Hedging is the use of financial instruments to reduce exposure to a specific risk.
Value at risk estimates the potential loss of a portfolio over a given period at a given confidence level.
Mergers and acquisitions involve the combination or purchase of companies.
A leveraged buyout is an acquisition financed largely with borrowed money.
Venture capital provides financing to early-stage companies with high growth potential.
Private equity funds invest in companies that are not publicly traded.
An annuity is a series of equal payments made at regular intervals.
A perpetuity is a stream of equal cash flows that continues indefinitely.
Amortizing loans repay both principal and interest through scheduled payments.
The loan-to-value ratio compares the size of a loan to the value of the asset securing it.
Collateral is an asset pledged by a borrower to secure a loan.
Insurance transfers risk from the policyholder to the insurer in exchange for a premium.
Behavioral finance studies how psychological biases influence financial decisions.
Loss aversion describes the tendency to feel losses more strongly than equivalent gains.
Payment services directives in the European Union regulate payment institutions and electronic payments.""".strip().split("\n")
pre=["","In finance, ","In economics and finance, ","From a financial perspective, ","As a general principle, "]
KEEP={"EBITDA","International","Generally","Modern","Beta","Know","Anti-money","Payment"}
KEEPW={"The European","The Basel","The Fisher","The Sharpe"}
for c in concepts:
    first=c.split()[0]; two=" ".join(c.split()[:2])
    low=c if (first in KEEP or two in KEEPW) else c[0].lower()+c[1:]
    for p in pre: put("concepts", p+low if p else c)

# ---------- worked numeric examples ----------
cur=["euros","dollars","pounds"]
def f2(x): return f"{x:,.2f}"
cnt=0
TARGET=4000
while cnt<TARGET-len(S)+cnt:
    kind=random.choice(["ci","si","pv","yield","pe","roe","growth","real","loan","cr","mcap","eps","dy"])
    c=random.choice(cur)
    if kind=="ci":
        P=random.choice(range(500,50001,500)); r=random.choice([1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8]); n=random.randint(2,30)
        FV=P*(1+r/100)**n
        s=random.choice([f"An investment of {P:,} {c} at {r}% annual interest compounded annually grows to about {f2(FV)} {c} after {n} years",
          f"With annual compounding at {r}%, {P:,} {c} becomes approximately {f2(FV)} {c} in {n} years",
          f"The future value of {P:,} {c} invested for {n} years at a {r}% annual rate, compounded yearly, is about {f2(FV)} {c}"])
    elif kind=="si":
        P=random.choice(range(1000,20001,250)); r=random.choice([2,3,4,5,6]); n=random.randint(1,10)
        I=P*r/100*n
        s=random.choice([f"At a simple interest rate of {r}% per year, a principal of {P:,} {c} earns {f2(I)} {c} in interest over {n} years",
          f"Simple interest on {P:,} {c} at {r}% for {n} years amounts to {f2(I)} {c}"])
    elif kind=="pv":
        FV=random.choice(range(1000,100001,1000)); r=random.choice([2,3,4,5,6,8,10]); n=random.randint(1,20)
        PV=FV/(1+r/100)**n
        s=random.choice([f"The present value of {FV:,} {c} received in {n} years, discounted at {r}% per year, is about {f2(PV)} {c}",
          f"Discounting {FV:,} {c} due in {n} years at an annual rate of {r}% gives a present value of approximately {f2(PV)} {c}"])
    elif kind=="yield":
        face=1000; cr=random.choice([2,3,4,5,6,7]); price=random.choice(range(850,1151,5))
        cy=cr*10/price*100
        s=random.choice([f"A bond with a face value of {face:,} {c}, a {cr}% annual coupon, and a price of {price:,} {c} has a current yield of about {cy:.2f}%",
          f"If a {cr}% coupon bond with a {face:,} {c} face value trades at {price:,} {c}, its current yield is approximately {cy:.2f}%"])
        s+=(". The bond trades at a premium" if price>1000 else ". The bond trades at a discount" if price<1000 else ". The bond trades at par")
    elif kind=="pe":
        p=random.choice(range(10,301)); e=round(random.uniform(0.5,15),2); pe=p/e
        s=random.choice([f"A share priced at {p} {c} with earnings per share of {e:.2f} {c} has a price-to-earnings ratio of about {pe:.1f}",
          f"If earnings per share are {e:.2f} {c} and the share price is {p} {c}, the P/E ratio is approximately {pe:.1f}"])
    elif kind=="roe":
        ni=random.choice(range(1,500))*100000; eq=ni*random.uniform(3,20); eq=round(eq,-5) or 100000
        s=random.choice([f"A company with net income of {ni:,} {c} and average equity of {eq:,.0f} {c} has a return on equity of {ni/eq*100:.1f}%",
          f"Net income of {ni:,} {c} on shareholders' equity of {eq:,.0f} {c} corresponds to a return on equity of about {ni/eq*100:.1f}%"])
    elif kind=="growth":
        a=random.choice(range(100,10001,50)); b=round(a*random.uniform(0.6,1.6))
        ch=(b-a)/a*100
        s=random.choice([f"If revenue changes from {a:,} to {b:,} thousand {c}, the percentage change is {ch:+.1f}%",
          f"An asset whose price moves from {a:,} to {b:,} {c} has a return of {ch:+.1f}%, excluding income"])
    elif kind=="real":
        i=random.choice([2,3,4,5,6,7,8,9,10]); pi=random.choice([1,1.5,2,2.5,3,4,5,6])
        rr=((1+i/100)/(1+pi/100)-1)*100
        s=random.choice([f"With a nominal return of {i}% and inflation of {pi}%, the real return is about {rr:.2f}% using the exact Fisher relation",
          f"A nominal interest rate of {i}% combined with {pi}% inflation implies a real interest rate of approximately {rr:.2f}%"])
    elif kind=="loan":
        P=random.choice(range(5000,400001,5000)); r=random.choice([2,3,3.5,4,4.5,5,6,7]); yrs=random.choice([3,5,10,15,20,25,30])
        m=r/1200; nn=yrs*12; pay=P*m/(1-(1+m)**-nn)
        s=random.choice([f"A loan of {P:,} {c} at a {r}% annual rate, repaid monthly over {yrs} years, has a monthly payment of about {f2(pay)} {c}",
          f"The monthly payment on a {yrs}-year amortizing loan of {P:,} {c} at {r}% annual interest is approximately {f2(pay)} {c}"])
    elif kind=="cr":
        ca=random.choice(range(100,5001,10)); cl=random.choice(range(100,5001,10))
        s=f"A company with current assets of {ca:,} thousand {c} and current liabilities of {cl:,} thousand {c} has a current ratio of {ca/cl:.2f}"
    elif kind=="mcap":
        p=round(random.uniform(1,500),2); sh=random.choice(range(1,1001))
        s=f"A company with {sh:,} million shares outstanding at a price of {p:.2f} {c} has a market capitalization of about {p*sh:,.1f} million {c}"
    elif kind=="eps":
        ni=random.choice(range(1,1000)); sh=random.choice(range(10,500))
        s=f"Net income of {ni:,} million {c} divided by {sh:,} million shares gives earnings per share of {ni/sh:.2f} {c}"
    else:
        d=round(random.uniform(0.1,10),2); p=random.choice(range(10,301))
        s=f"An annual dividend of {d:.2f} {c} on a share priced at {p} {c} gives a dividend yield of {d/p*100:.2f}%"
    n0=len(S); put("calculations",s); cnt+=len(S)-n0

base=[x for x in S if x[0]!="comparative"]; comp=[x for x in S if x[0]=="comparative"]
random.shuffle(comp)
rows=base+comp[:10000-len(base)]
print(len(base),len(comp),len(rows))
random.shuffle(rows)
with open(__import__("pathlib").Path(__file__).with_name("finance_sentences.csv"),"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["id","category","sentence"])
    for i,(c,s) in enumerate(rows,1): w.writerow([i,c,s])
