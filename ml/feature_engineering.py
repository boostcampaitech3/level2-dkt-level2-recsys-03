import pandas as pd


def get_user_correct_answer(df): 
    """유저마다 시간순으로 정답 수를 누적하여 계산"""
    return df.groupby("userID")["answerCode"].transform(lambda x: x.cumsum().shift(1))


def get_user_total_answer(df):
    """유저마다 시간순으로 문제 풀이 수를 누적하여 계산 (제일 앞 시간대는 NaN으로 채움)"""
    return df.groupby("userID")["answerCode"].cumcount()


def get_user_acc(df):
    """유저마다 시간순으로 정답률을 계속 계산 (제일 앞 시간대는 NaN으로 채움)
    get_user_correct_answer, get_user_total_answer를 실행해야 fucntion 실행 가능
    """
    return df["user_correct_answer"]/df["user_total_answer"]


def get_future(df):
    """미래 특정 시점에 문제 정답을 맞췄는지 못 맞췄는 지에 대한 여부(1,2 step 적용, 빈 시간대는 NaN으로 채움)"""
    edu_shift_df = df.copy()
    step_2 = edu_shift_df.groupby("userID")["answerCode"].shift(-2)
    step_1 = edu_shift_df.groupby("userID")["answerCode"].shift(-1)

    return step_2, step_1


def get_past(df):
    """과거 특정 시점에 문제 정답을 맞췄는지 못 맞췄는 지에 대한 여부(1,2 step 적용, 빈 시간대는 NaN으로 채움)"""
    edu_shift_df = df.copy()
    step_2 = edu_shift_df.groupby("userID")["answerCode"].shift(2)
    step_1 = edu_shift_df.groupby("userID")["answerCode"].shift(1)

    return step_2, step_1


def get_time(df):
    """유저가 문제별로 풀이에 사용한 시간(초단위로 계산)"""
    diff = df.loc[:, ["userID", "Timestamp"]].groupby("userID").diff().fillna(pd.Timedelta(seconds=0))
    diff = diff.fillna(pd.Timedelta(seconds=0))
    
    return diff["Timestamp"].apply(lambda x: x.total_seconds())    


def get_future_correct(df):
    """유저별 미래에 맞출 문제 수(유저의 총 문제 맞출 수에서 시간이 지날수록 차감)"""
    reversed_edu_correct_df = df.iloc[::-1].copy()
    reversed_edu_correct_df["shift"] = reversed_edu_correct_df.groupby("userID")["answerCode"].shift().fillna(0)
    reversed_edu_correct_df["future_correct"] = reversed_edu_correct_df.groupby("userID")["shift"].cumsum()
    tmp_df = reversed_edu_correct_df.iloc[::-1]
    
    return tmp_df["future_correct"]


def get_past_content_correct(df):
    """유저가 현재 풀고 있는 문제를 과거에 맞춘 횟수"""
    edu_correct_df = df.copy()
    edu_correct_df['shift'] = edu_correct_df.groupby(["userID", "assessmentItemID"])["answerCode"].shift().fillna(0)

    return edu_correct_df.groupby(["userID", "assessmentItemID"])["shift"].cumsum()


def get_past_content_count(df):
    """유저별 문제를 과거에 푼 횟수"""
    edu_correct_df = df.copy()

    return edu_correct_df.groupby(["userID", "assessmentItemID"]).cumcount()


def get_average_content_correct(df):
    """유저별 과거 문제를 풀었을 때 정답률
    get_past_content_correct, get_average_content_correct를 실행해야 function 실행 가능"""
    edu_correct_df = df.copy()

    return (edu_correct_df["past_content_correct"] / edu_correct_df["past_content_count"]).fillna(0)


def get_mean_time(df):
    """이동 평균(Rolling Mean)을 사용해 현재 푸는 문제를 포함하여 최근 3개 문제의 평균 풀이시간
    get_time을 실행하여 function 실행 가능"""
    edu_rolling_df = df.copy()

    return edu_rolling_df.groupby(['userID'])['time'].rolling(3).mean().values


def get_time_median(df):
    """유저별 문제 풀이에 사용한 시간의 중간값
    get_time을 실행해야 function 실행 가능"""
    edu_agg_df = df.copy()
    # 중간값 (median)
    agg_df = edu_agg_df.groupby("userID")["time"].agg(["median"])
    # mapping을 위해 pandas DataFrame을 dictionary형태로 변환
    agg_dict = agg_df.to_dict()
    # 구한 통계량을 각 사용자에게 mapping

    return edu_agg_df["userID"].map(agg_dict["median"])


def get_hour(df):
    """유저가 문제를 푸는 시간대(hour)"""
    edu_time_df = df.copy()

    return edu_time_df["Timestamp"].transform(lambda x: pd.to_datetime(x, unit="s").dt.hour)


def get_hour_mode(df):
    """유저가 문제를 푸는 시간대(유저의 주 활동 시간)
    get_hour을 실행해야 function 실행 가능"""
    edu_time_df = df.copy()
    mode_dict = edu_time_df.groupby(["userID"])["hour"].agg(lambda x: pd.Series.mode(x)[0]).to_dict()
    
    return edu_time_df["userID"].map(mode_dict)
    

def get_normalized_time(df):
    """문제 풀이에 사용한 시간 정규화"""
    edu_custom_df = df.copy()
    # time만 transform

    return edu_custom_df.groupby("userID")["time"].transform(lambda x: (x - x.mean())/x.std())


def get_relative_time(df):
    """문제풀이에 사용한 시간과 중간값 차이를 통해 문제풀이 시간을 상대적으로 비교
    get_time을 실행해야 function 실행 가능"""
    edu_custom_df = df.copy()
    # apply를 사용해 time column을 직접 지정할 수 있다

    return edu_custom_df.groupby("userID").apply(lambda x: x["time"] - x["time"].median()).values


def get_etc(df):
    """merge는 마지막에 한번에 계산하기 때문에 7가지 feature 생성함수를 한꺼번에 묶음
    correct_per_hour : 유저의 시간대별 정답률
    test_mean, test_sum : testId별 정답 평균, 합
    tag_mean, tag_sum : KnowledgeTag별 정답 평균, 합
    assess_mean, assess_sum " assessmentItemID별 평균, 합
    get_hour을 실행해야 function 실행 가능"""
    edu_time_df = df.copy()
    hour_dict = edu_time_df.groupby(["hour"])["answerCode"].mean().to_dict()
    edu_time_df["correct_per_hour"] = edu_time_df["hour"].map(hour_dict)
    correct_per_time = edu_time_df.groupby(["hour"])[["correct_per_hour"]].mean()

    correct_t = df.groupby(["testId"])["answerCode"].agg(["mean", "sum"])
    correct_t.columns = ["test_mean", "test_sum"]
    correct_k = df.groupby(["KnowledgeTag"])["answerCode"].agg(["mean", "sum"])
    correct_k.columns = ["tag_mean", "tag_sum"]
    correct_a = df.groupby(["assessmentItemID"])["answerCode"].agg(["mean", "sum"])
    correct_a.columns = ["assess_mean", "assess_sum"]
    correct_u = df.groupby(["userID"])["answerCode"].agg(["mean", "sum"])
    correct_u.columns = ["user_mean", "user_sum"]

    df = pd.merge(df, correct_t, on=["testId"], how="left")
    df = pd.merge(df, correct_k, on=["KnowledgeTag"], how="left")
    df = pd.merge(df, correct_a, on=["assessmentItemID"], how="left")
    df = pd.merge(df, correct_u, on=["userID"], how="left")
    df = pd.merge(df, correct_per_time, on=["hour"], how="left")

    return df

    
def get_features(df): 
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=["userID","Timestamp"], inplace=True)

    df["user_correct_answer"] = get_user_correct_answer(df)
    df["user_total_answer"] = get_user_total_answer(df)
    df["user_acc"] = get_user_acc(df)

    df["correct_shift_-2"], df["correct_shift_-1"] = get_future(df)
    df["correct_shift_2"], df["correct_shift_1"] = get_past(df)

    df["time"] = get_time(df)
    df["future_correct"] = get_future_correct(df)

    df["past_content_correct"] = get_past_content_correct(df)
    df["past_content_count"] = get_past_content_count(df)
    df["average_content_correct"] = get_average_content_correct(df)

    df["mean_time"] = get_mean_time(df)
    df["time_median"] = get_time_median(df)

    df["hour"] = get_hour(df)
    df["hour_mode"] = get_hour_mode(df)
    df["normalized_time"] = get_normalized_time(df)
    df["relative_time"] = get_relative_time(df)

    return get_etc(df)