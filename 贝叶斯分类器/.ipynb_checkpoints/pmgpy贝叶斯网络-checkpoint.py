import numpy as np
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling

alarm_model = get_example_model("alarm")
samples = BayesianModelSampling(alarm_model).forward_sample(size=int(1e5))
samples.head()

from pgmpy.models import BayesianNetwork

model_struct = BayesianNetwork(ebunch=alarm_model.edges())
model_struct.nodes()



# Fitting the model using Maximum Likelihood Estimator

from pgmpy.estimators import MaximumLikelihoodEstimator

mle = MaximumLikelihoodEstimator(model=model_struct, data=samples)

# Estimating the CPD for a single node.
print(mle.estimate_cpd(node="FIO2"))
print(mle.estimate_cpd(node="CVP"))

# Estimating CPDs for all the nodes in the model
mle.get_parameters()[:10]  # Show just the first 10 CPDs in the output



# Verifying that the learned parameters are almost equal.
np.allclose(
    alarm_model.get_cpds("FIO2").values, mle.estimate_cpd("FIO2").values, atol=0.01
)

