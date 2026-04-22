
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import re

def count_email_domains(df: pd.DataFrame) -> Dict[str, int]:
    #TODO
    result={}
    for email in df['email'].dropna():
        email=str(email).strip().lower()
        if "@" in email:
            domain = email.split("@")[-1]
            result[domain]=result.get(domain,0)+1
    return result
def registration_timeline(df: pd.DataFrame) -> Tuple[str, str, List[int], List[int]]:
    #TODO
    dates =pd.to_datetime(df['registration_date']).dropna()
    months=dates.dt.strftime("%Y-%m")
    month_counts=months.value_counts().sort_index()

    if len(month_counts)==0:
        return "","",[],[]
    min_date=month_counts.index[0]
    max_date=month_counts.index[-1]

    period_range=pd.period_range(start=min_date,end=max_date,freq="M")
    full_months=[str(p) for p in period_range]
    counts=[int(month_counts.get(month,0)) for month in full_months]
    timeline=list(range(len(full_months)))
    return min_date,max_date,timeline,counts
def count_jobs(df: pd.DataFrame) -> Dict[str, int]:
    #TODO
    result={}
    for job in df['job'].dropna():
        job=str(job).strip()
        result[job]=result.get(job,0)+1
    return result
def analyze_introductions(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, int]]:
    #TODO
    word_counter=Counter()
    length_counter=Counter()
    for intro in df['introduction'].dropna():
        inrto=str(intro)
        words=re.findall(r"\b\w+\b",intro.lower())
        word_counter.update(words)
        length_counter[len(intro)]+=1
    top5_words=dict(word_counter.most_common(5))
    intro_lengths=dict(sorted(length_counter.items()))

    return top5_words,intro_lengths

        
def visualize_results(email_stats: Dict[str, int],
                      job_stats: Dict[str, int],
                      min_date: str,
                      max_date: str,
                      timeline: List[int],
                      counts: List[int],
                      word_stats: Dict[str, int],
                      intro_lengths: Dict[int, int]):

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("User Data Analysis Results", fontsize=16)

    axs[0, 0].bar(email_stats.keys(), email_stats.values(), color='skyblue')
    axs[0, 0].set_title("Email Domain Distribution")
    axs[0, 0].tick_params(axis='x', rotation=45)

    axs[0, 1].bar(job_stats.keys(), job_stats.values(), color='orange')
    axs[0, 1].set_title("Job Title Distribution")
    axs[0, 1].tick_params(axis='x', rotation=45)

    axs[1, 0].scatter(timeline, counts, c='green')
    axs[1, 0].set_title(f"Registration Timeline Distribution\n{min_date} to {max_date}")
    axs[1, 0].set_xlabel("Registration Year-Month (YYYYMM)")
    axs[1, 0].set_ylabel("Number of Users")

    axs[1, 1].bar(word_stats.keys(), word_stats.values(), color='purple')
    axs[1, 1].set_title("Most Frequent Words in Introduction")
    axs[1, 1].tick_params(axis='x', rotation=45)

    axs[2, 0].bar(intro_lengths.keys(), intro_lengths.values(), color='brown')
    axs[2, 0].set_title("Introduction Length Distribution")
    axs[2, 0].set_xlabel("Length")
    axs[2, 0].set_ylabel("Frequency")

    axs[2, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('visualize.png')


def main():
    df = pd.read_csv("users.csv")
    # 各项统计
    email_stats = count_email_domains(df)
    min_date, max_date, timeline, counts = registration_timeline(df)
    job_stats = count_jobs(df)
    word_stats, intro_lengths = analyze_introductions(df)

    # 可视化展示
    visualize_results(email_stats, job_stats, min_date, max_date, timeline, counts, word_stats, intro_lengths)

if __name__ == "__main__":
    main()
