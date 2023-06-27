"""NOT USED, PURE FOR INITIAL ANALYSIS"""

clear;
%% 1) Damage regression without restrictions
start_year = 1980;
delete_years = {};
Data = readtable("Data/Regression data.xlsx", 'Sheet', 'Data', ReadVariableNames=true);
columns = Data.Properties.VariableNames;
delete_columns = columns(46: start_year - 1900 + 44);
for y = 1:length(delete_years)
    delete_columns(end+1) = delete_years(y);
end
for c = 1:length(delete_columns)
    col = char(delete_columns(c));
    Data = Data(~(Data.(col) == 1), :);
    Data = removevars(Data, {col});
end
for i= 1: width(Data)
    Data.(i)(isnan(Data.(i))) = 0;
end
% Delete variables
Data = removevars(Data, ["TBSR"]);
final_columns = Data.Properties.VariableNames;
mdl = fitlm(Data, 'ResponseVar', 'GDP');
coefficients = mdl.Coefficients
AIC = mdl.ModelCriterion.AIC
%% 2) Damage regression without temperature squared and precipitation
start_year = 1980;
delete_years = {};
Data = readtable("Data/Regression data.xlsx", 'Sheet', 'Data', ReadVariableNames=true);
columns = Data.Properties.VariableNames;
delete_columns = columns(46: start_year - 1900 + 44);
for y = 1:length(delete_years)
    delete_columns(end+1) = delete_years(y);
end
for c = 1:length(delete_columns)
    col = char(delete_columns(c));
    Data = Data(~(Data.(col) == 1), :);
    Data = removevars(Data, {col});
end
for i= 1: width(Data)
    Data.(i)(isnan(Data.(i))) = 0;
end
% Delete variables
Data = removevars(Data, ["Precipitation", "TBSR", "Temperature_2"]);
final_columns = Data.Properties.VariableNames;
mdl = fitlm(Data, 'ResponseVar', 'GDP');
coefficients = mdl.Coefficients
AIC = mdl.ModelCriterion.AIC
%% 3) Damage regression without year effects, precipitation
start_year = 1980;
Data = readtable("Data/Regression data.xlsx", 'Sheet', 'Data', ReadVariableNames=true);
columns = Data.Properties.VariableNames;
delete_columns = columns(46: start_year - 1900 + 44);
for c = 1:length(delete_columns)
    col = char(delete_columns(c));
    Data = Data(~(Data.(col) == 1), :);
    Data = removevars(Data, {col});
end
Data = Data(:, 1:end-(2023-start_year));
Data = removevars(Data, ["TBSR", "Precipitation", "MeanSeaLevel_2"]);
%% 
for i= 1: width(Data)
    Data.(i)(isnan(Data.(i))) = 0;
end
% Delete variables
final_columns = Data.Properties.VariableNames;
mdl = fitlm(Data, 'ResponseVar', 'GDP');
coefficients = mdl.Coefficients
AIC = mdl.ModelCriterion.AIC
R2 = mdl.Rsquared
%% 4) Damage regression without year effects, precipitation, and temperature squared
start_year = 1980;
delete_years = {};
Data = readtable("Data/Regression data.xlsx", 'Sheet', 'Data', ReadVariableNames=true);
columns = Data.Properties.VariableNames;
delete_columns = columns(46: start_year - 1900 + 44);
for y = 1:length(delete_years)
    delete_columns(end+1) = delete_years(y);
end
for c = 1:length(delete_columns)
    col = char(delete_columns(c));
    Data = Data(~(Data.(col) == 1), :);
    Data = removevars(Data, {col});
end
for i= 1: width(Data)
    Data.(i)(isnan(Data.(i))) = 0;
end
% Delete variables
Data = removevars(Data, ["TBSR", "Precipitation", "Temperature_2", "MeanSeaLevel_2"]);
Data = Data(:, 1:end-(2023-start_year));
final_columns = Data.Properties.VariableNames;
mdl = fitlm(Data, 'ResponseVar', 'GDP');
coefficients = mdl.Coefficients
AIC = mdl.ModelCriterion.AIC
%% 5) Damage regression without year effects, precipitation, and economic variables
start_year = 1980;
delete_years = {};
Data = readtable("Data/Regression data.xlsx", 'Sheet', 'Data', ReadVariableNames=true);
columns = Data.Properties.VariableNames;
delete_columns = columns(46: start_year - 1900 + 44);
for y = 1:length(delete_years)
    delete_columns(end+1) = delete_years(y);
end
for c = 1:length(delete_columns)
    col = char(delete_columns(c));
    Data = Data(~(Data.(col) == 1), :);
    Data = removevars(Data, {col});
end
for i= 1: width(Data)
    Data.(i)(isnan(Data.(i))) = 0;
end
% Delete variables
Data = removevars(Data, ["TBSR", "Precipitation", "UNEM", "CPI", "NGLR", "EQ", "MeanSeaLevel_2"]);
Data = Data(:, 1:end-(2023-start_year));
final_columns = Data.Properties.VariableNames;
mdl = fitlm(Data, 'ResponseVar', 'GDP');
coefficients = mdl.Coefficients
AIC = mdl.ModelCriterion.AIC
R2 = mdl.Rsquared
%% 6) Damage regression without year effects, precipitation, temperature_2, and economic variables
start_year = 1980;
delete_years = {};
Data = readtable("Data/Regression data.xlsx", 'Sheet', 'Data', ReadVariableNames=true);
columns = Data.Properties.VariableNames;
delete_columns = columns(46: start_year - 1900 + 44);
for y = 1:length(delete_years)
    delete_columns(end+1) = delete_years(y);
end
for c = 1:length(delete_columns)
    col = char(delete_columns(c));
    Data = Data(~(Data.(col) == 1), :);
    Data = removevars(Data, {col});
end
for i= 1: width(Data)
    Data.(i)(isnan(Data.(i))) = 0;
end
% Delete variables
Data = removevars(Data, ["TBSR", "Precipitation", "Temperature_2", "UNEM", "CPI", "NGLR", "EQ"]);
Data = Data(:, 1:end-(2023-start_year));
final_columns = Data.Properties.VariableNames;
mdl = fitlm(Data, 'ResponseVar', 'GDP');
coefficients = mdl.Coefficients
AIC = mdl.ModelCriterion.AIC
%% 7) Damage regression without country + year effects, precipitation
start_year = 1980;
delete_years = {};
Data = readtable("Data/Regression data.xlsx", 'Sheet', 'Data', ReadVariableNames=true);
columns = Data.Properties.VariableNames;
delete_columns = columns(46: start_year - 1900 + 44);
for y = 1:length(delete_years)
    delete_columns(end+1) = delete_years(y);
end
for c = 1:length(delete_columns)
    col = char(delete_columns(c));
    Data = Data(~(Data.(col) == 1), :);
    Data = removevars(Data, {col});
end
for i= 1: width(Data)
    Data.(i)(isnan(Data.(i))) = 0;
end
% Delete variables
Data = removevars(Data, ["TBSR", "Precipitation", "MeanSeaLevel_2"]);
Data = Data(:, 1:8);
final_columns = Data.Properties.VariableNames;
mdl = fitlm(Data, 'ResponseVar', 'GDP');
coefficients = mdl.Coefficients
AIC = mdl.ModelCriterion.AIC
R2 = mdl.Rsquared
%% 8) Damage regression without year effects, precipitation, and economic variables
start_year = 1980;
delete_years = {};
Data = readtable("Data/Regression data.xlsx", 'Sheet', 'Data', ReadVariableNames=true);
columns = Data.Properties.VariableNames;
delete_columns = columns(46: start_year - 1900 + 44);
for y = 1:length(delete_years)
    delete_columns(end+1) = delete_years(y);
end
for c = 1:length(delete_columns)
    col = char(delete_columns(c));
    Data = Data(~(Data.(col) == 1), :);
    Data = removevars(Data, {col});
end
for i= 1: width(Data)
    Data.(i)(isnan(Data.(i))) = 0;
end
% Delete variables
Data = removevars(Data, ["TBSR", "Precipitation", "MeanSeaLevel_2", "UNEM", "CPI", "NGLR", "EQ"]);
Data = Data(:, 1:4);
final_columns = Data.Properties.VariableNames;
mdl = fitlm(Data, 'ResponseVar', 'GDP');
coefficients = mdl.Coefficients
AIC = mdl.ModelCriterion.AIC
R2 = mdl.Rsquared
%% 9) Damage regression without year effects, precipitation with Temperature interaction
start_year = 1980;
delete_years = {};
Data = readtable("Data/Regression data.xlsx", 'Sheet', 'Data TA interaction', ReadVariableNames=true);
columns = Data.Properties.VariableNames;
delete_columns = columns(148: start_year - 1900 + 146);
for y = 1:length(delete_years)
    delete_columns(end+1) = delete_years(y);
end
for c = 1:length(delete_columns)
    col = char(delete_columns(c));
    Data = Data(~(Data.(col) == 1), :);
    Data = removevars(Data, {col});
end
for i= 1: width(Data)
    Data.(i)(isnan(Data.(i))) = 0;
end
% Delete variables
Data = removevars(Data);
Data = Data(:, 1:end-(2023-start_year));
final_columns = Data.Properties.VariableNames;
mdl = fitlm(Data, 'ResponseVar', 'GDP');
coefficients = mdl.Coefficients
AIC = mdl.ModelCriterion.AIC
R2 = mdl.Rsquared
%% Final model
Data = readtable("Data/Regression data.xlsx", 'Sheet', 'Regression data', ReadVariableNames=true);
% Delete variables
Data = removevars(Data, ["Temperature_2"]);
final_columns = Data.Properties.VariableNames;
mdl = fitlm(Data, 'ResponseVar', 'GDP');
coefficients = mdl.Coefficients
AIC = mdl.ModelCriterion.AIC
R2 = mdl.Rsquared
%% Tests
% Residuals
residuals = mdl.Residuals.Raw;
% Statistics
avg = mean(residuals);
stdev = sqrt(var(residuals));
t_test = avg/stdev;
skew = skewness(residuals);
kurt = kurtosis(residuals);
correlation_tests = [];
r = [];
for i=1:length(final_columns)
    pearson = corrcoef(residuals, table2array(Data(:, i)));
    r(i) = pearson(1, 2);
    correlation_tests(i) = r(i) / sqrt((1-r(i)^2)^2/(size(Data(:,1), 1)-1));
end
% Normality test
[jb_h, jb_pvalue] = jbtest(residuals);
% Autocorrelation Ljung-Box test
[lbq_h, lbq_pvalue] = lbqtest(residuals);
% Heteroskedasticity
bp_pvalue = TestHet(residuals, table2array(Data(:, :)), '-BPK');
% Archtest heteroskedasticity
[eng_h, eng_pvalue] = archtest(residuals);
%% Newey-West SE
[cov_hac, se_hac, coefficients_hac] = hac(mdl);
