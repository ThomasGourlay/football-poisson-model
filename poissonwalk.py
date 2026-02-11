import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import poisson
from scipy.optimize import minimize
from dataclasses import dataclass

# ------------------------------------------------------------------------------
# defining data types
# ------------------------------------------------------------------------------

@dataclass
class Team:
    '''
    stores some basic information about a team
    '''
    name: str
    attack: float
    defence: float

@dataclass
class Settings:
    '''
    store the settings (hyperparameters) used
    sd_movement = sd of change in team abilities between seasons
        (higher -> team ratings are allowed to change more between seasons)
    h_adv_init = initial home advantage. this just serves as 
    '''
    sd_movement: float
    h_adv_init: float



# ------------------------------------------------------------------------------
# preprocessing and formatting functions
# ------------------------------------------------------------------------------

def make_date(datestr: str) -> date:
    '''
    reformat date string DD/MM/YYYY into datetime date

    '''
    info = datestr.split("/")
    day = int(info[0])
    month = int(info[1])
    year = int(info[2])
    return date(year, month, day)

def calculate_season(date: date, season_start: int) -> int:
    '''
    figure out what season we are in (2023/24 season is called 2024, etc)
    only makes sense for sports that have seasons split by july
    '''
    if date.month >= season_start:
        return date.year
    return date.year - 1

def gather_matches(match_data: pd.DataFrame, team_idx: dict) -> np.ndarray:
    '''
    takes a dataframe of matches and converts it into a np array with just basic information
    Returns:
        np.ndarray: array of tuples of matches (home index, away index, home goals, away goals)
    '''
    match_data = match_data.copy()
    match_data["h_idx"] = match_data["HomeTeam"].apply(lambda x: team_idx[x])
    match_data["a_idx"] = match_data["AwayTeam"].apply(lambda x: team_idx[x])

    match_data = match_data[["h_idx", "a_idx", "FTHG", "FTAG"]]
    matches = [tuple(i) for i in match_data.itertuples(index=False, name=None)]

    return np.array(matches)

def theta_to_df(big_theta: np.ndarray, season_matches: dict, team_idx: dict, n_teams: int) -> pd.DataFrame:
    """
    Convert a flat theta vector into a DataFrame of team ratings per season.

    Args:
        big_theta (np.ndarray): flat array containing attack/defence ratings
            for all seasons, with home advantage as the final element.
        season_matches (dict): dict only used to infer
            season ordering.
        teams (list of str): team names in index order
        n_teams (int): tumber of teams

    Returns:
        pd.DataFrame: team ratings with columns ["Season", "Team", "Attack",
        "Defence"], plus a final row for home advantage.
    """
    home_adv = big_theta[-1]
    rows = []
    teams = list(team_idx.keys())
    # ensure consistent season ordering and get the seasons
    sorted_seasons = sorted(season_matches.keys())

    for season_idx, season in enumerate(sorted_seasons):
        start = 2 * n_teams * season_idx
        end = 2 * n_teams * (season_idx + 1)
        season_theta = big_theta[start:end]

        attacks = season_theta[:n_teams]
        defences = season_theta[n_teams:]

        for i, team in enumerate(teams):
            rows.append({
                "Season": season,
                "Team": team,
                "Attack": attacks[i],
                "Defence": defences[i]
            })

    # add home advantage
    rows.append({
        "Season": "All",
        "Team": "HomeAdv",
        "Attack": home_adv,
        "Defence": np.nan
    })

    return pd.DataFrame(rows)


def collect_data(filename: str, season_start: int):

    print(f'Collecting data from {filename}')
    match_data = pd.read_csv(filename)
    match_data["Date"] = match_data["Date"].apply(make_date)            # reformat date

    match_data["Season"] = match_data["Date"].apply(calculate_season, args=(season_start,))   # figure out what season we're in

    match_data.sort_values(by="Date",inplace=True)                      # sort data by date
    match_data = match_data.reset_index(drop=True)

    first = match_data.iloc[0]["Date"]
    last = match_data.iloc[-1]["Date"]
    print(f'First match observed: {first}')
    print(f'Last match observed: {last}')

    # Get unique teams from BOTH home and away
    home_teams = set(match_data["HomeTeam"].unique())
    away_teams = set(match_data["AwayTeam"].unique())
    teams = sorted(list(home_teams | away_teams))       # Union of both sets
    team_idx = {j:i for i,j in enumerate(teams)}

    n_teams = len(team_idx)
    print(f"We found {n_teams} total teams. They were {list(team_idx.keys())}")

    seasons = dict(tuple(match_data.groupby("Season"))) # split the data into seasons
    n_seasons = len(seasons)
    
    season_matches = {                                  # group the data into seasons and make them lists of tuples 
        season: gather_matches(df, team_idx)
        for season, df in seasons.items()
    }
    print("We found the following seasons:")
    for i in season_matches.keys():
        print(f"{i}/{i+1} with {len(season_matches[i])} matches")
    print("I hope that sounds right!")
    return match_data, season_matches, n_seasons, team_idx, n_teams



# ------------------------------------------------------------------------------
# mathematical functions and optimisation
# ------------------------------------------------------------------------------

def neg_log_season(theta: np.ndarray, matches_array: np.ndarray, n_teams: int) -> float:
    '''
    calculates the negative log likelihood of a season, given attack, defence, home adv ratings
    vectorised for decent performance.

    Parameters:
        theta (np.ndarray) attack, defence, ratings and home advantage
            first n elements are attack ratings, then defence ratings, then home adv.
            this has to be a flat array because this function is minimised.
        matches_array (np.ndarray) array of matches in the format returned by gather_matches
        n_teams (int) number of teams. used purely to split the theta array easily
    Returns:
        float: negative logarithm of the probability of this season occuring, given the
            ratings. this calculates P(X|θ)
    '''
    attacks = theta[:n_teams]
    defences = theta[n_teams:2*n_teams]
    home_adv = theta[-1]

    attacks = attacks - np.mean(attacks)  # identifiability by enforcing attacks mean=0

    h_idx = matches_array[:, 0].astype(int)
    a_idx = matches_array[:, 1].astype(int)
    h_goals = matches_array[:, 2].astype(int)
    a_goals = matches_array[:, 3].astype(int)

    # calculate lambda param for home and away poisson distributinos
    lam_h = np.exp(home_adv + attacks[h_idx] - defences[a_idx])
    lam_h = np.clip(lam_h, 1e-10, 1e2)  # prevent underflow / overflow (usually unnecessary)

    lam_a = np.exp(-home_adv + attacks[a_idx] - defences[h_idx])
    lam_a = np.clip(lam_a, 1e-10, 1e2)  # prevent underflow / overflow (usually unnecessary)

    # calculate log likelihood
    log_like = poisson.logpmf(h_goals, lam_h) + poisson.logpmf(a_goals, lam_a)
    return -np.sum(log_like)


def log_likelihood(big_theta: np.ndarray, seasons_matches: dict, n_teams: int, settings: Settings) -> float:
    '''
    big log likelihood function
    see documentation for explanation. essentially MAP (MLE with prior)

    Parameters:
        big_theta (np.ndarray) similar to theta from neg_log_likelihood,
            but contains ratings for every season, followed by home advantage
        seasons_matches (dict) a dict where key = year and value = season
            a season is an array as returned by gather_matches
        n_teams (int) number of teams for splitting theta
        settings (Settings) settings bruddah
    '''
    # take theta and break it into components (season by seasons ratings + home adv)
    unique_seasons = list(seasons_matches.keys())
    n_seasons = len(unique_seasons)
    thetas=dict()
    home_adv = big_theta[-1]
    for season in range(n_seasons):
        curr_theta = big_theta[2*n_teams*season:2*n_teams*(season+1)]
        curr_theta = np.append(curr_theta, home_adv)
        thetas[unique_seasons[season]] = curr_theta

    # for each season, calculate (negative) log-likelihood and add to total
    nll = 0
    for season in seasons_matches.keys():
        matches = seasons_matches[season]
        theta = thetas[season]
        nll += neg_log_season(theta=theta, matches_array=matches, n_teams=n_teams)


    # now calculate nll of change in rating between seasons
    # stack all season ratings into array (n_seasons, n_teams)
    attack_ratings = np.stack([big_theta[2*n_teams*season:2*n_teams*season + n_teams] for season in range(n_seasons)], axis=0)
    defence_ratings = np.stack([big_theta[2*n_teams*season + n_teams : 2*n_teams*(season+1)] for season in range(n_seasons)], axis=0)

    # find differences across seasons
    attack_diffs = attack_ratings[1:] - attack_ratings[:-1]  # shape (n_seasons-1, n_teams)
    defence_diffs = defence_ratings[1:] - defence_ratings[:-1]

    # log-likelihood contribution from random walk
    sd = settings.sd_movement
    n = (n_seasons - 1) * n_teams

    # vectorised
    rw_ll = -0.5 * n * np.log(2*np.pi) - n * np.log(sd) - np.sum(attack_diffs**2) / (2*sd**2)
    rw_ll += -0.5 * n * np.log(2*np.pi) - n * np.log(sd) - np.sum(defence_diffs**2) / (2*sd**2)

    # subtract because function returns nll
    nll -= rw_ll
    return nll

def fit_params(season_matches: dict, n_teams: int, n_seasons: int, settings: Settings, team_idx: dict):
    '''
    Fit team ratings using match data and settings

    Parameters:
        season_matches (dict) dictionary of year: list/array of matches as from gather_matches function
        n_teams (int) number of teams
        n_seasons (int) numb of seasons
        settings (Settings) settings for the model fitting
        teams (list) list of teams in index order
            note: this should probably be streamlined to just use the team_idx dictionary
    Return:
        pd.DataFrame: a df of the ratings
    '''
    theta0 = np.zeros(2*n_seasons*n_teams + 1)
    theta0[-1] = settings.h_adv_init

    res=minimize(
        fun=log_likelihood,
        x0=theta0,
        args=(season_matches,n_teams,settings),
        method="L-BFGS-B"
    )
    ratings=res.x
    ratings = theta_to_df(ratings, season_matches, team_idx, n_teams)
    return ratings


# ------------------------------------------------------------------------------
# functions for after fitting data, calculate probability, backtesting
# ------------------------------------------------------------------------------

def p_total_over(goals: float, home_team: Team, away_team: Team, home_adv: float) -> float:
    '''
    calculates the probability of the total goals in a match exceeding some number
    Parameters:
        goals (float) number of goals we want to find prob of exceeding. 
            typically X.5 or something
        home_team (Team) the home team in the match
        away_team (Team) the away team in the match
        home_adv (float) the size of the home advantage in this league
    Returns:
        float: probability
    '''
    lam_h = np.exp(home_adv + home_team.attack - away_team.defence)
    lam_a = np.exp(-home_adv + away_team.attack - home_team.defence)
    lam = lam_h + lam_a
    prob = 1 - poisson.cdf(np.floor(goals), lam)
    return prob

def p_over(goals: float, home_team: Team, away_team: Team, home_adv: float, home_or_away: str) -> float:
    '''
    calculates the probability of one team's goals in a match exceeding some number
    Parameters:
        goals (float) number of goals we want to find prob of exceeding. 
            typically X.5 or something
        home_team (Team) the home team in the match
        away_team (Team) the away team in the match
        home_adv (float) the size of the home advantage in this league
        home_or_away (str) "home"->calculate P(home exceeds goals), "away"->calculate P(away exceeds goals)
    Returns:
        float: probability
    '''
    if not ((home_or_away=="home") or (home_or_away=="away")):
        raise ValueError("home_or_away should be \"home\" or \"away\"")
    
    if home_or_away == "home":
        lam = np.exp(home_adv + home_team.attack - away_team.defence)
        return 1 - poisson.cdf(np.floor(goals), lam)
    lam = np.exp(-home_adv + away_team.attack - home_team.defence)
    return 1 - poisson.cdf(np.floor(goals), lam)

def backtest(df_ratings: pd.DataFrame, match_data: pd.DataFrame, init_pnl: float =0) -> pd.DataFrame:
    '''
    a backtest that uses attacking/defensive ratings and match data to bet on
    over/under 2.5 goals, using Bet365 odds as our book

    uses a constant 1 unit bet size for simplicity, may add kelly system if it seems helpful later

    results are only as reliable as the data given 

    this function is not very optimised, so takes up to a few seconds to run. could be optimised if needed
    Parameters:
        df_ratings (pd.DataFrame) a dataframe containing ratings, cols ["Team", "Attack", "Defence"]
            also should contain a team called "HomeAdv" which is the home advantage
        match_data (pd.DataFrame) a dataframe containing matches, should be in date order
        init_pnl (float) initial money to start with. this is useful when we're doing
            multiple backtests using different info and want to see how they all perform long term etc
    Result:
        pd.DataFrame: contains all net pnl after each bet placed on each date. often there are multiple
            entries for one date, because we made multiple bets on that date. doesnt really matter
    '''
    money = init_pnl
    n_bets = 0
    dates = []
    pnl = []

    # get the home advantage from the df
    home_adv = df_ratings.loc[df_ratings["Team"]=="HomeAdv", "Attack"].iloc[0]
    for _, match in match_data.iterrows():
        # get important information about the match
        season = match["Season"]
        date = match["Date"]
        curr_ratings = df_ratings[df_ratings["Season"]==season]

        # get important information about the teams
        h_name = match["HomeTeam"]
        h_att = curr_ratings.loc[curr_ratings["Team"]==h_name, "Attack"].iloc[0]
        h_def = curr_ratings.loc[curr_ratings["Team"]==h_name, "Defence"].iloc[0]

        a_name = match["AwayTeam"]
        a_att = curr_ratings.loc[curr_ratings["Team"]==a_name, "Attack"].iloc[0]
        a_def = curr_ratings.loc[curr_ratings["Team"]==a_name, "Defence"].iloc[0]

        # package the teams into Team dataclass
        home_team = Team(h_name, h_att, h_def)
        away_team = Team(a_name, a_att, a_def)

        # check if the over bet is underpriced, if it is, bet
        fair = 1/p_total_over(2.5, home_team=home_team, away_team=away_team, home_adv=home_adv)
        odds = match["B365>2.5"]
        if fair < odds:

            total = match["FTHG"] + match["FTAG"]
            if total > 2.5:
                money += odds - 1
            else:
                money -= 1
            

        # check if the under bet is underpriced, if it is, bet
        fair = 1/(1-1/fair)
        odds = match["B365<2.5"]
        if fair < odds:
            total = match["FTHG"] + match["FTAG"]
            if total < 2.5:
                money += odds - 1
            else:
                money -= 1
        pnl.append(money)
        dates.append(date)

    df = pd.DataFrame({
        "pnl": pnl,
        "date": dates
    })
    return df


def fit_and_backtest(match_data: pd.DataFrame, settings: Settings, team_idx: dict, fit_begin: date, test_begin: date, test_end: date, init_pnl: float=0):
    '''
    fit data from fit_begin, until (not including) test_begin
    backtest from test_begin until test_end
    test_end should be in the same season as test_begin
    Parameters: cbs explaining
    '''
    n_teams = len(team_idx)

    match_data = match_data.copy()

    # allocate train and test data

    train = match_data[(match_data["Date"] < test_begin) & (match_data["Date"]>=fit_begin)]
    test = match_data[(match_data["Date"] >= test_begin) & (match_data["Date"] <= test_end)]
    seasons = dict(tuple(train.groupby("Season")))
    season_matches = {season: gather_matches(df, team_idx)
                      for season, df in seasons.items()}
    n_seasons = len(seasons)
    print(f"fitting paramaters on data from {fit_begin} up to {test_begin}")

    ratings = fit_params(season_matches, n_teams, n_seasons, settings, team_idx)
    print(f"testing params on data from {test_begin} until {test_end}")
    results = backtest(ratings, test, init_pnl)
    final_pnl = results.iloc[-1]["pnl"]
    net = final_pnl-init_pnl
    print(f"in this period we made net ${net:.2f}\n")
    return results, final_pnl

def second_half_backtest(match_data, settings, team_idx, first, last, season_start):
    '''
    requires revision doesnt work
    '''
    mylist=[]
    pnl=0
    for year in range(first,last+1):
        f_begin = date(year-2,season_start,1)
        t_start = date(year, 1, 1)
        t_end = date(year,6,30)
        results, pnl = fit_and_backtest(match_data, settings, team_idx, f_begin, t_start, t_end, pnl)
        mylist.append(results)
    final = pd.concat(mylist)
    return final