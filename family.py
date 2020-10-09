import torch


class Family():
    '''
    Create Family class, currently only 4 distributions are available:
        - 'Normal': bernoulli distribution with logits (identity)
        - 'Poisson': poisson with rate (exp)
        - 'Bernoulli': bernoulli distribution with logits (identity)
        - 'Bernoulli_prob': bernoulli distribution with probabilities (sigmoid)
        - 'Multinomial': multinomial distribution parameterized by total_count(=1) and logits
        - 'Multinomial_prob': multinomial distribution parameterized by total_count(=1) and probs
    Later more distributions will be implemented, such as Gamma, Beta and NegativeBinomial
    
    Parameters
    ----------
        family : string
            The current distribution defined by the user
    Attributes
    -------
        families: dictionary
            A dictionary holding all the available distributions as keys and values are again dictionaries with the parameters as keys and values the formula which applies for each parameter
        family: string
            The current distribution defined by the user
    '''
    def __init__(self,family):        
        self.families = {'Normal':['loc', 'scale'], 
                       'Poisson': ['rate'], 
                       'Bernoulli': ['logits'],
                       'Bernoulli_prob':['probs'],
                       'Multinomial_prob':['probs']}
#                        'Multinomial':['logits'],
#                        'Gamma':['concentration', 'rate'],
#                        'Beta':['concentration1', 'concentration0'],
                       #'NegativeBinomial':['loc', 'scale']}  # available family list
        assert family in self.families.keys(),'Given distribution is not available. Please try with a different distribution. Available distributions are %s' % (self.families.keys())

        self.family = family   # current distribution family

    def get_params(self):
        '''
        Return parameters for current distribution family, e.g. 'rate' for 'Poisson' family
        '''
        return self.families[self.family]
        
    def get_distribution_layer_type(self):   
        '''
        Return corresponding type of distributional layer, which is used in the network 
        '''
        if self.family == "Normal":
            distribution_layer_type = torch.distributions.normal.Normal
        elif self.family == "Poisson":
            distribution_layer_type = torch.distributions.poisson.Poisson
        elif self.family == "Bernoulli" or self.family == "Bernoulli_prob":
            distribution_layer_type = torch.distributions.bernoulli.Bernoulli       
        elif self.family == "Multinomial_prob":
            distribution_layer_type = torch.distributions.multinomial.Multinomial 
#         elif family == "Gamma":
#             distribution_layer_type = torch.distributions.gamma.Gamma
#         elif family == "Beta":
#             distribution_layer_type = torch.distributions.beta.Beta
#         elif family == "NegativeBinomial":
#             distribution_layer_type = torch.distributions.negative_binomial.NegativeBinomial
        else:
            raise ValueError('Unknown distribution')
           
        return distribution_layer_type
        
    def get_distribution_trafos(self, pred):
        '''
        Do transformation for each parameter
        Parameters
        ----------
        pred : tensor
            The output of each single_parameter_net
        Returns
        -------
        pred_trafo: tensor
            Transformed output
        '''
        pred_trafo = dict()
        add_const = 1e-8
        
        if self.family == "Normal":
            pred_trafo["loc"] = pred["loc"]
            pred_trafo["scale"] = add_const + pred["scale"].exp()
            
        elif self.family == "Poisson":
            pred_trafo["rate"] = add_const + pred["rate"].exp()
            
        elif self.family == "Bernoulli":
            pred_trafo["logits"] = pred["logits"]
            
        elif self.family == "Bernoulli_prob":
            pred_trafo["probs"] = torch.nn.functional.sigmoid(pred["probs"])
            
        elif self.family == "Multinomial_prob":
            pred_trafo["total_count"] = 1
            pred_trafo["probs"] = torch.nn.functional.softmax(pred["probs"])
            
#         elif self.family == "Multinomial":
#             pred_trafo["total_count"] = 1
#             pred_trafo["logits"] = torch.nn.functional.softmax(pred["probs"])

#         elif family == "Gamma":
#             pred_trafo["concentration"] = add_const + pred["concentration"].exp()
#             pred_trafo["rate"] = add_const + pred["rate"].exp()
            
#         elif family == "Beta":
#             pred_trafo["concentration1"] = add_const + pred["concentration1"].exp()
#             pred_trafo["concentration0"] = add_const + pred["concentration0"].exp()
            
#         elif family == "NegativeBinomial":   
#             ####### to do: loc, scale -> f(total count) , p(probs)
#             pred_trafo["total_count"] = pred["total_count"]  # constant
#             pred_trafo["probs"] = pred["probs"]
            
        else:
            raise ValueError('Unknown distribution')
                 
        return pred_trafo