import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from tests import pyunit_utils

# The purpose of this test to make sure that constraint GLM works in the presence of collinear columns in the dataset
def test_constraints_collinear_columns():
    # first two columns are enums, the last 4 are real columns
    h2o_data = pyunit_utils.genTrainFrame(10000, 6, enumCols=2, enumFactors=2, responseLevel=2, miscfrac=0, randseed=12345)
    # create extra collinear columns
    num1 = h2o_data[2]*0.2-0.5*h2o_data[3]
    num2 = -0.8*h2o_data[4]+0.1*h2o_data[5]
    h2o_collinear = num1.cbind(num2)
    h2o_collinear.set_names(["col1", "col2"])
    train_data = h2o_data.cbind(h2o_collinear)
    y = "response"
    x = train_data.names
    x.remove(y)
    bc = []

    name = "AGE"
    lower_bound = 0.1
    upper_bound = 0.5
    bc.append([name, lower_bound, upper_bound])

    name = "RACE"
    lower_bound = -0.5
    upper_bound = 0.5
    bc.append([name, lower_bound, upper_bound])

    name = "DCAPS"
    lower_bound = -0.4
    upper_bound = 0.4
    bc.append([name, lower_bound, upper_bound])

    name = "DPROS"
    lower_bound = -0.3
    upper_bound = 0.3
    bc.append([name, lower_bound, upper_bound])

    name = "PSA"
    lower_bound = -0.2
    upper_bound = 0.5
    bc.append([name, lower_bound, upper_bound])

    name = "VOL"
    lower_bound = -0.5
    upper_bound = 0.5
    bc.append([name, lower_bound, upper_bound])

    name = "GLEASON"
    lower_bound = -0.5
    upper_bound = 0.5
    bc.append([name, lower_bound, upper_bound])

    beta_constraints = h2o.H2OFrame(bc)
    beta_constraints.set_names(["names", "lower_bounds", "upper_bounds"])

#    h2o_glm = H2OGeneralizedLinearEstimator(family="binomial", nfolds=10, alpha=0.5, beta_constraints=beta_constraints)
    h2o_glm = H2OGeneralizedLinearEstimator(family="binomial", compute_p_values=True, lambda_=0.0, solver="irlsm")
    h2o_glm.train(x=x, y=y, training_frame=train_data )

    print("Done")



if __name__ == "__main__":
    pyunit_utils.standalone_test(test_constraints_collinear_columns)
else:
    test_constraints_collinear_columns()
