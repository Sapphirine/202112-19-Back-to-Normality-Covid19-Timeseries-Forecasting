import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
from sklearn.preprocessing import MinMaxScaler,StandardScaler

pd.options.mode.chained_assignment = None

us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}
    
# invert the dictionary
abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))

def produce_sample(case_df_by_state):
    data_array = None
    for state in case_states:
        if state in vacc_states:
            this_past_df = pd.DataFrame({})

            this_case_df = case_df_by_state[[state]]
            this_case_df.rename(columns={state:'case_ma'},inplace=True)
            this_vacc_df = vacc_df_by_state[[state]]
            this_vacc_df.rename(columns={state:'vacc_rate'},inplace=True)
            this_df = pd.concat([this_case_df,this_vacc_df],axis=1)

            # this_df.to_csv('./data_by_state/'+state+'.csv',index=True)

            this_past_df['case_ma'] = this_df.case_ma
            this_past_df['vacc_rate'] = this_df.vacc_rate

            for i in range(1,21):
                this_past_df['_'.join(['case',str(i),'day_ago'])] = this_df.case_ma.shift(periods=i)
            for i in range(1,21):
                this_past_df['_'.join(['vacc',str(i),'day_ago'])] = this_df.vacc_rate.shift(periods=i)
            this_past_df.dropna(inplace=True)

            if data_array is None:
                data_array = this_past_df.to_numpy()
            else:
                data_array = np.concatenate((data_array,this_past_df.to_numpy()),axis=0)
    
    X = data_array[:,2:]
    y = data_array[:,0]
    y =  y.reshape(-1,1)

    return X,y


if __name__ == "__main__":
    url="https://static.usafacts.org/public/data/covid-19/covid_confirmed_usafacts.csv?_ga=2.135985718.681365941.1640147350-1281724743.1639242218"
    s=requests.get(url).content
    case_df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    url="https://static.usafacts.org/public/data/external/COVID19_CDC_Vaccination_CSV_Download.csv?_ga=2.232519448.681365941.1640147350-1281724743.1639242218"
    s=requests.get(url).content
    vacc_df=pd.read_csv(io.StringIO(s.decode('utf-8')))

    case_states = list(pd.value_counts(case_df.State).keys())
    case_df_by_state = pd.DataFrame({})
    for state in case_states:
        this_df = case_df[case_df.State == state]
        this_df = this_df.iloc[:,4:]
        this_ts = this_df.sum(axis=0).diff(periods=1).rolling(window=7).mean()
        this_df = this_ts.to_frame(name=abbrev_to_us_state[state])
        case_df_by_state = pd.concat([case_df_by_state,this_df],axis=1)

    vacc_df = vacc_df[vacc_df.GEOGRAPHY_LEVEL == 'State']
    vacc_df.sort_values(['GEOGRAPHY_NAME','DATE'], ascending=True,inplace=True)

    vacc_df_by_state = pd.DataFrame({})
    vacc_states = list(pd.value_counts(vacc_df.GEOGRAPHY_NAME).keys())
    for state in vacc_states:
        this_df = vacc_df[vacc_df.GEOGRAPHY_NAME == state]
        this_df = this_df[['DATE','FULLY_VACCINATED_PERCENT']]
        this_df.set_index(keys=['DATE'],inplace=True)
        this_df.rename(columns={'FULLY_VACCINATED_PERCENT':state},inplace=True)
        vacc_df_by_state = pd.concat([vacc_df_by_state,this_df],axis=1)
    vacc_df_by_state.sort_index(axis=0,inplace=True)

    vacc_i, vacc_f = tuple(np.array(list(vacc_df_by_state.index))[[1,-1]])
    case_i, case_f = tuple(np.array(list(case_df_by_state.index))[[1,-1]])
    case_states = case_df_by_state.columns.tolist()
    case_df_by_state_minmax = pd.DataFrame(MinMaxScaler().fit_transform(case_df_by_state), columns=case_df_by_state.columns,index=case_df_by_state.index)
    case_df_by_state_standa = pd.DataFrame(StandardScaler().fit_transform(case_df_by_state), columns=case_df_by_state.columns,index=case_df_by_state.index)
    X, y = produce_sample(case_df_by_state_minmax)
    np.save('./feature_minmax.npy',X)
    np.save('./target_minmax.npy', y)