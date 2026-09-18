import csv, random
random.seed(11)
FIRST=["Anna","Marco","Laura","David","Sofia","Jonas","Elena","Mikko","Priya","Lucas","Hannah","Pablo","Aino","Tomás","Clara","Yusuf","Nora","Felix","Irene","Samuel","Leila","Oskar","Marta","Daniel","Chiara","Ravi","Julia","Erik","Inés","Kenji"]
LAST=["Virtanen","García","Rossi","Müller","Novak","Silva","Korhonen","Martin","Fischer","López","Bianchi","Nieminen","Dubois","Kowalski","Andersen","Moreau","Ferreira","Lindqvist","Costa","Weber"]
CO=["Nordwave Systems","Brightpath Analytics","Kivi Labs","Sierra Cloud","Helix Data","Arcturus Software","Bluefjord Tech","Lumen Retail","Orbita Media","Pinecrest Health","Vertex Logistics","Quanta Finance"]
PROD=["DataPulse","CloudNest","TaskFlow","InsightHub","SecureGate","SyncBoard","MetricLens","PayStream"]
DAYS=["Monday","Tuesday","Wednesday","Thursday","Friday"]
TIMES=["9:00","10:30","11:00","13:00","14:00","15:30","16:00"]
MONTHS=["January","February","March","April","May","June","September","October","November"]
def name(): return random.choice(FIRST), random.choice(LAST)
def greet(fn, formal):
    return random.choice([f"Dear {fn},",f"Hello {fn},"] if formal else [f"Hi {fn},",f"Hello {fn},",f"Hey {fn},"])
def close(fn,ln,role,co,formal):
    c=random.choice(["Best regards,","Kind regards,","Sincerely,"] if formal else ["Best,","Thanks,","Cheers,","Best regards,"])
    return f"{c}\n{fn} {ln}\n{role}, {co}" if formal else f"{c}\n{fn}"
def date(): return f"{random.choice(DAYS)}, {random.randint(1,28)} {random.choice(MONTHS)}"

CS_ROLES=["Software Engineer","Backend Developer","DevOps Engineer","Engineering Manager","Data Engineer","Security Engineer","Tech Lead","QA Engineer","Machine Learning Engineer"]
MK_ROLES=["Marketing Manager","Content Strategist","Growth Lead","Brand Manager","Social Media Manager","Product Marketing Manager","Campaign Coordinator"]
SERVICES=["authentication service","payment API","search service","notification service","data pipeline","user dashboard","reporting module","mobile backend"]
BUGS=["a memory leak that causes the service to restart every few hours","a race condition when two users update the same record","incorrect time zone handling in scheduled jobs","a null pointer exception when the user profile is incomplete","slow queries caused by a missing database index","an expired TLS certificate on the staging environment"]
TECH=["Kubernetes","PostgreSQL","Kafka","Terraform","React","Rust","GraphQL","Redis"]
LANG=["Python","Go","TypeScript","Java","Rust"]

def rows_cs():
    out=[]
    # 1 bug report
    for _ in range(140):
        fn,ln=name(); rf,rl=name(); co=random.choice(CO); svc=random.choice(SERVICES); bug=random.choice(BUGS); v=f"{random.randint(1,4)}.{random.randint(0,12)}.{random.randint(0,9)}"
        formal=random.random()<0.4
        instr=f"Write {'a formal' if formal else 'an'} email to {rf}, a colleague, reporting a bug in the {svc} (version {v}): {bug}."
        body=random.choice([
f"""I wanted to let you know that we found a bug in the {svc}, version {v}. The issue is {bug}.

We first noticed it in the logs yesterday and were able to reproduce it in the staging environment. So far it affects a limited number of requests, but it could become more serious under higher load.

I have opened a ticket with the reproduction steps and the relevant log excerpts. Could you take a look when you have a moment and let me know if you need more details?""",
f"""Quick heads-up: there is a bug in version {v} of the {svc}. It turns out to be {bug}.

Steps to reproduce, logs, and a short analysis are in the ticket. I suspect the problem was introduced in the latest release, so a rollback might be a reasonable short-term option.

Would you be able to review it today? I'm happy to pair on the fix if that helps."""])
        out.append(("computer science","bug report",instr,"",f"Subject: Bug in {svc} {v}\n\n{greet(rf,formal)}\n\n{body}\n\n{close(fn,ln,random.choice(CS_ROLES),co,formal)}"))
    # 2 code review request
    for _ in range(110):
        fn,ln=name(); rf,rl=name(); co=random.choice(CO); lang=random.choice(LANG); svc=random.choice(SERVICES); pr=random.randint(100,4999)
        feat=random.choice(["adds retry logic with exponential backoff","refactors the database access layer","adds unit tests for the billing logic","introduces caching for frequent read queries","migrates the configuration to environment variables","adds structured logging"])
        instr=f"Write an email asking {rf} to review pull request #{pr}, which {feat} in the {svc}."
        body=f"""Could you review pull request #{pr} when you get a chance? It {feat} in the {svc}.

The change is written in {lang} and touches about {random.randint(3,25)} files. The main logic is in the service layer; the rest are tests and small updates to call sites. All tests pass locally and in CI.

I would especially appreciate your feedback on error handling and naming. If possible, I'd like to merge it before {random.choice(DAYS)}."""
        out.append(("computer science","code review request",instr,"",f"Subject: Review request: PR #{pr}\n\n{greet(rf,False)}\n\n{body}\n\n{close(fn,ln,'',co,False)}"))
    # 3 incident update
    for _ in range(110):
        fn,ln=name(); co=random.choice(CO); svc=random.choice(SERVICES); dur=random.choice([12,25,40,55,75,90]); cause=random.choice(["a misconfigured load balancer rule","a failed database migration","an exhausted connection pool","a faulty deployment that was rolled back","a DNS configuration change","an expired API credential"])
        instr=f"Write a formal incident summary email to the engineering team about a {dur}-minute outage of the {svc} caused by {cause}."
        body=f"""This email summarizes the incident that affected the {svc} on {date()}.

Impact: The service was unavailable or degraded for approximately {dur} minutes. Users experienced failed requests and timeouts during this period.

Root cause: The outage was caused by {cause}.

Resolution: The on-call engineer identified the issue, applied a fix, and confirmed that the service had fully recovered. Monitoring shows normal error rates and latency since then.

Next steps: We will hold a blameless post-mortem this week, add an alert that would have detected the problem earlier, and update the runbook. Action items will be tracked in the incident ticket.

Please reply to this email if you have questions or additional information."""
        out.append(("computer science","incident report",instr,"",f"Subject: Incident summary: {svc} outage\n\nDear team,\n\n{body}\n\n{close(fn,ln,random.choice(['Engineering Manager','Site Reliability Engineer','Tech Lead']),co,True)}"))
    # 4 meeting scheduling
    for _ in range(110):
        fn,ln=name(); rf,rl=name(); co=random.choice(CO); t=random.choice(TECH); d1,d2=random.sample(DAYS,2); t1,t2=random.sample(TIMES,2)
        topic=random.choice([f"the migration to {t}",f"the architecture of the new {random.choice(SERVICES)}","the sprint planning for next quarter",f"performance issues in {t}","the security audit findings"])
        formal=random.random()<0.5
        instr=f"Write an email to {rf} proposing a meeting to discuss {topic}, offering two time slots."
        body=f"""I'd like to schedule a {random.choice([30,45,60])}-minute meeting to discuss {topic}. There are a few open technical decisions, and it would be good to align before we move forward.

Would either of these slots work for you?
- {d1} at {t1}
- {d2} at {t2}

If neither suits you, feel free to suggest another time. I'll send a calendar invite with a short agenda once we agree."""
        out.append(("computer science","meeting request",instr,"",f"Subject: Meeting about {topic}\n\n{greet(rf,formal)}\n\n{body}\n\n{close(fn,ln,random.choice(CS_ROLES),co,formal)}"))
    # 5 deployment / maintenance announcement
    for _ in range(90):
        fn,ln=name(); co=random.choice(CO); svc=random.choice(SERVICES); d=date(); t=random.choice(TIMES); h=random.choice([1,2,3])
        instr=f"Write an email announcing scheduled maintenance of the {svc} on {d} at {t} UTC, lasting {h} hour{'s' if h>1 else ''}."
        body=f"""We will perform scheduled maintenance on the {svc} on {d}, starting at {t} UTC. The maintenance window is expected to last up to {h} hour{'s' if h>1 else ''}.

During this time, the service may be intermittently unavailable. We are upgrading the underlying infrastructure to improve reliability and performance.

No action is required from you. If you have critical jobs scheduled during this window, please reschedule them or contact us in advance. We will send a confirmation once the maintenance is complete."""
        out.append(("computer science","maintenance announcement",instr,"",f"Subject: Scheduled maintenance: {svc} on {d}\n\nDear all,\n\n{body}\n\n{close(fn,ln,random.choice(['DevOps Engineer','Platform Team Lead','Site Reliability Engineer']),co,True)}"))
    # 6 security notice
    for _ in range(70):
        fn,ln=name(); co=random.choice(CO); lib=random.choice(["a logging library","an XML parser","an image processing library","an HTTP client library","a JSON serialization library"])
        instr=f"Write an email to developers asking them to update dependencies after a vulnerability was disclosed in {lib}."
        body=f"""A security vulnerability has been disclosed in {lib} that several of our services depend on. The issue could allow an attacker to execute unexpected code under certain conditions.

Please take the following steps by the end of {random.choice(DAYS)}:
1. Check whether your service uses the affected library, directly or as a transitive dependency.
2. Update it to the patched version listed in the security ticket.
3. Run the test suite and deploy the change through the normal release process.

The security team has already scanned our repositories, and the list of affected services is attached to the ticket. If you are unsure whether your service is affected, contact the security team and we will help you verify it."""
        out.append(("computer science","security notice",instr,"",f"Subject: Action required: vulnerability in {lib}\n\nDear developers,\n\n{body}\n\n{close(fn,ln,'Security Engineer',co,True)}"))
    # 7 onboarding welcome
    for _ in range(70):
        fn,ln=name(); rf,rl=name(); co=random.choice(CO); t1,t2=random.sample(TECH,2)
        instr=f"Write a friendly welcome email to {rf}, a new developer joining the team, with first-week information."
        body=f"""Welcome to the team! We're very glad to have you with us at {co}.

Here's what your first week will look like:
- On your first day, IT will help you set up your laptop and accounts.
- You'll get access to our repositories and the developer documentation.
- We'll walk you through our stack, which is built mainly on {t1} and {t2}.
- By the end of the week, you'll pick up a small starter task to get familiar with the codebase and review process.

{random.choice(FIRST)} will be your onboarding buddy, so don't hesitate to ask them anything. See you on {random.choice(DAYS)}!"""
        out.append(("computer science","onboarding",instr,"",f"Subject: Welcome to the team, {rf}!\n\n{greet(rf,False)}\n\n{body}\n\n{close(fn,ln,'Engineering Manager',co,False)}"))
    # 8 replies: technical question
    for _ in range(140):
        fn,ln=name(); rf,rl=name(); co=random.choice(CO); t=random.choice(TECH); svc=random.choice(SERVICES)
        q,a=random.choice([
          ("why our API responses got slower after the last release","The slowdown comes from a new database query added in the last release that runs without an index. Adding a composite index on the filtered columns should bring response times back to normal, and I've prepared a migration for it."),
          ("whether we should use REST or GraphQL for the new client","For this client I'd lean toward REST, since the data needs are simple and our existing tooling, caching, and monitoring are built around it. GraphQL would make more sense if the client needed flexible queries across many related resources."),
          (f"how to get access to the {t} staging cluster","Access is managed through the internal access portal. Request the staging developer role there, and once your manager approves it, the credentials will be available within about an hour."),
          ("whether the nightly backup job is working","Yes, the job has completed successfully every night this week. I also ran a restore test on Tuesday, and the data was recovered without errors."),
          (f"when the {svc} will support the new API version","We plan to support the new API version in the next release, scheduled in about three weeks. Both versions will run in parallel for at least two months so clients have time to migrate."),
        ])
        incoming=f"Subject: Question\n\nHi {fn},\n\nCould you tell me {q}?\n\nThanks,\n{rf}"
        instr=f"Reply to the following email from {rf} answering the question."
        body=random.choice(["Thanks for reaching out.","Good question.","Thanks for your message."])+" "+a+"\n\nLet me know if anything is unclear or if you need more details."
        out.append(("computer science","reply to question",instr,incoming,f"Subject: Re: Question\n\n{greet(rf,False)}\n\n{body}\n\n{close(fn,ln,'',co,False)}"))
    # 9 replies: request for deadline extension / decline politely
    for _ in range(90):
        fn,ln=name(); rf,rl=name(); co=random.choice(CO); feat=random.choice(["the export feature","the new login flow","the API documentation","the load testing report","the data migration"])
        incoming=f"Subject: Can we deliver {feat} by Friday?\n\nHi {fn},\n\nThe client is asking whether we can deliver {feat} by this Friday instead of next month. Is that realistic?\n\nBest,\n{rf}"
        instr=f"Reply to {rf} explaining politely that the earlier deadline for {feat} is not realistic, and propose an alternative."
        body=f"""Thanks for checking with me first. Unfortunately, delivering {feat} by Friday isn't realistic without cutting testing and code review, which would put quality and stability at risk.

What we could do instead is deliver a limited first version with the most important functionality by the end of next week, followed by the complete version on the original schedule. That way the client gets something useful early, and we keep our quality standards.

If that works for you, I can outline exactly what the first version would include so you can share it with the client."""
        out.append(("computer science","reply declining request",instr,incoming,f"Subject: Re: Can we deliver {feat} by Friday?\n\n{greet(rf,False)}\n\n{body}\n\n{close(fn,ln,'',co,False)}"))
    return out

AUD=["small business owners","HR managers","online retailers","IT decision-makers","freelance designers","restaurant owners","university students","healthcare administrators"]
BEN=["save several hours of manual work every week","get a clear view of their key metrics in one place","automate repetitive reporting","collaborate with their team in real time","reduce errors in their billing process","launch campaigns faster"]
def rows_mk():
    out=[]
    # launch
    for _ in range(140):
        fn,ln=name(); co=random.choice(CO); p=random.choice(PROD); aud=random.choice(AUD); ben=random.choice(BEN); disc=random.choice([10,15,20,25,30])
        instr=f"Write a product launch marketing email for {p}, aimed at {aud}, highlighting that it helps them {ben} and offering a {disc}% launch discount."
        body=random.choice([
f"""We're excited to introduce {p}, a new tool built for {aud} who want to {ben}.

With {p}, you can:
- set up your workspace in minutes, with no technical skills required
- connect the tools you already use
- see results from the very first week

To celebrate the launch, we're offering {disc}% off your first year for customers who sign up before the end of the month.

Start your free trial today and see how much time you can save.""",
f"""What if you could {ben}?

That's exactly why we built {p}. After talking with hundreds of {aud}, we designed a simple solution that fits the way you already work.

For a limited time, new customers get {disc}% off. There's no long-term commitment, and you can cancel at any time.

Try {p} free for 14 days and let us know what you think."""])
        subj=random.choice([f"Introducing {p}: {disc}% off for early customers",f"Meet {p}, built for {aud}",f"{p} is here"])
        out.append(("marketing","product launch",instr,"",f"Subject: {subj}\n\nHello,\n\n{body}\n\n{close(fn,ln,random.choice(MK_ROLES),co,True)}"))
    # newsletter
    for _ in range(100):
        fn,ln=name(); co=random.choice(CO); p=random.choice(PROD); m=random.choice(MONTHS)
        t1,t2=random.sample(["a new dashboard layout","faster data exports","two-factor authentication","a mobile app update","new integrations with popular tools","improved search"],2)
        instr=f"Write a monthly customer newsletter email for {p} for {m}, announcing {t1} and {t2}."
        body=f"""Here's what's new in {p} this {m}.

New: {t1[0].upper()+t1[1:]}
Based on your feedback, we've released {t1}. It's available to all users starting today.

Improved: {t2[0].upper()+t2[1:]}
We've also added {t2}, making your daily work smoother.

From the blog
Read our latest guide with practical tips on getting more value from {p}.

As always, we'd love to hear from you. Just reply to this email with your ideas or questions."""
        out.append(("marketing","newsletter",instr,"",f"Subject: {p} {m} update: {t1} and more\n\nHi there,\n\n{body}\n\n{close(fn,ln,'The '+p+' Team',co,False).replace(chr(10)+fn,chr(10)+'The '+p+' Team')}"))
    # webinar invite
    for _ in range(100):
        fn,ln=name(); sf,sl=name(); co=random.choice(CO); aud=random.choice(AUD); d=date(); t=random.choice(TIMES)
        topic=random.choice(["growing your customer base with email marketing","using data to make better business decisions","building a content strategy that converts","automating your sales workflow","improving customer retention"])
        instr=f"Write an email inviting {aud} to a free webinar on {topic} on {d} at {t} CET, presented by {sf} {sl}."
        body=f"""Join us for a free live webinar on {topic}.

Date: {d}
Time: {t} CET
Speaker: {sf} {sl}

In 45 minutes, you'll learn practical strategies that {aud} can apply right away, see real examples, and get your questions answered in a live Q&A.

Seats are limited, so reserve your spot today. Can't attend live? Register anyway and we'll send you the recording."""
        out.append(("marketing","webinar invitation",instr,"",f"Subject: Free webinar: {topic}\n\nHello,\n\n{body}\n\n{close(fn,ln,random.choice(MK_ROLES),co,True)}"))
    # cold outreach B2B
    for _ in range(110):
        fn,ln=name(); rf,rl=name(); co=random.choice(CO); tco=random.choice([c for c in CO if c!=co]); p=random.choice(PROD); ben=random.choice(BEN)
        instr=f"Write a short, polite B2B cold outreach email to {rf} {rl} at {tco}, introducing {p}."
        body=f"""I'm {fn} from {co}. I noticed that {tco} has been growing quickly, and I thought {p} might be relevant for your team.

{p} helps companies like yours {ben}. One of our customers reduced the time spent on this work by about a third within three months.

Would you be open to a 20-minute call next week to see whether it could be a good fit? If this isn't a priority right now, just let me know and I won't follow up."""
        out.append(("marketing","cold outreach",instr,"",f"Subject: Helping {tco} {ben}\n\n{greet(rf,True)}\n\n{body}\n\n{close(fn,ln,'Account Executive',co,True)}"))
    # follow-up
    for _ in range(80):
        fn,ln=name(); rf,rl=name(); co=random.choice(CO); p=random.choice(PROD)
        instr=f"Write a brief follow-up email to {rf} after sending a proposal for {p} last week with no reply."
        body=f"""I wanted to follow up on the proposal for {p} that I sent last week. I know things get busy, so I'm simply checking whether you had a chance to review it.

If you have any questions about pricing, implementation, or timelines, I'd be happy to answer them. I can also adjust the proposal if your priorities have changed.

Would a short call on {random.choice(DAYS)} be helpful?"""
        out.append(("marketing","follow-up",instr,"",f"Subject: Following up on the {p} proposal\n\n{greet(rf,True)}\n\n{body}\n\n{close(fn,ln,'Account Manager',co,True)}"))
    # re-engagement
    for _ in range(70):
        fn,ln=name(); co=random.choice(CO); p=random.choice(PROD); disc=random.choice([15,20,30])
        instr=f"Write a re-engagement email to inactive users of {p} offering {disc}% off if they come back."
        body=f"""It's been a while since you last used {p}, and we wanted to reach out.

A lot has changed since your last visit: we've made {p} faster, easier to use, and added features many of you asked for.

To welcome you back, here's {disc}% off your next three months. Just log in and the discount will be applied automatically.

If {p} is no longer right for you, we'd genuinely appreciate hearing why, so we can keep improving."""
        out.append(("marketing","re-engagement",instr,"",f"Subject: We miss you: {disc}% off to come back\n\nHi there,\n\n{body}\n\n{close(fn,ln,'Customer Success',co,False).replace(chr(10)+fn,chr(10)+'The '+p+' Team')}"))
    # replies: customer complaint
    for _ in range(110):
        fn,ln=name(); rf,rl=name(); co=random.choice(CO); p=random.choice(PROD)
        prob,fix=random.choice([
          ("I was charged twice for my subscription this month.","I've checked your account and confirmed the duplicate charge. We have issued a full refund for the second payment, which should appear on your statement within 5 to 7 business days."),
          ("The export feature has not worked for three days.","Our team identified an issue affecting exports for some accounts and deployed a fix this morning. Could you try again and let me know whether it works for you now?"),
          ("I never received the discount promised in your email.","You're right, the discount should have been applied. I've added it to your account manually, and it will be reflected on your next invoice."),
          ("Your support team has not answered my ticket for a week.","That is not the level of service we aim for, and I'm sorry for the delay. I've taken over your ticket personally and will make sure it's resolved.")])
        incoming=f"Subject: Problem with {p}\n\nHello,\n\n{prob} This is very frustrating. Please fix it as soon as possible.\n\n{rf} {rl}"
        instr=f"Reply professionally and empathetically to this customer complaint from {rf} {rl}."
        body=f"""Thank you for contacting us, and I'm sorry for the trouble you've experienced with {p}.

{fix}

We appreciate your patience and your feedback, which helps us improve. If there is anything else I can do, please reply to this email and I will personally follow up."""
        out.append(("marketing","reply to complaint",instr,incoming,f"Subject: Re: Problem with {p}\n\n{greet(rf,True)}\n\n{body}\n\n{close(fn,ln,'Customer Support Specialist',co,True)}"))
    # replies: partnership inquiry
    for _ in range(70):
        fn,ln=name(); rf,rl=name(); co=random.choice(CO); tco=random.choice([c for c in CO if c!=co])
        incoming=f"Subject: Partnership opportunity\n\nHi {fn},\n\nI'm {rf} from {tco}. We think our audiences overlap and would like to explore a co-marketing partnership, such as a joint webinar or guest articles. Would you be interested?\n\nBest,\n{rf} {rl}"
        instr=f"Reply positively to {rf}'s partnership proposal and suggest next steps."
        body=f"""Thanks for reaching out, and great to hear from {tco}. A co-marketing partnership sounds interesting, and I agree that our audiences have a lot in common.

A joint webinar could be a good place to start, since it lets us test the collaboration with a clear scope. Guest articles could follow if it goes well.

Would you be available for a 30-minute call next week to discuss goals, topics, and timelines? Feel free to suggest a time that works for you."""
        out.append(("marketing","reply to partnership",instr,incoming,f"Subject: Re: Partnership opportunity\n\n{greet(rf,False)}\n\n{body}\n\n{close(fn,ln,random.choice(MK_ROLES),co,False)}"))
    return out

rows=rows_cs()+rows_mk()
seen=set(); uniq=[]
for r in rows:
    k=(r[2],r[4])
    if k not in seen: seen.add(k); uniq.append(r)
random.shuffle(uniq)
def fmt(instr,inc,email):
    s="### Instruction:\n"+instr+"\n\n"
    if inc: s+="### Incoming email:\n"+inc+"\n\n"
    return s+"### Email:\n"+email+"\n<|endoftext|>"
with open(__import__("pathlib").Path(__file__).with_name("email_finetune.csv"),"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["id","domain","email_type","instruction","incoming_email","email","text"])
    for i,(d,t,ins,inc,em) in enumerate(uniq,1): w.writerow([i,d,t,ins,inc,em,fmt(ins,inc,em)])
print(len(rows),len(uniq))
