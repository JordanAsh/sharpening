import torch






class GaltonWatson_Generator():

    def __init__(self, parent_probs, child_probs, separate_children = True):
        """
        Synthetic GW data generator.

        Args:

            parent_probs: list
                A list of probabilities for each parent node. Must be nonnegative and sum to one.
            child_probs:
                A list of lists of probabilities for each child node. Must be nonnegative and sum to one.
            separate_children:
                A boolean indicating whether to treat each child node as a separate token or if the child nodes can only be decoded in the context of their parent node.
        """

        assert len(parent_probs) == len(child_probs), "The number of types of parents must match the number of parent distributions."

        self.parent_distibution = torch.distributions.categorical.Categorical(torch.tensor(parent_probs))

        child_distributions, num_children = [], []
        for probs in child_probs:
            child_distributions.append(torch.distributions.categorical.Categorical(torch.tensor(probs)))
            num_children.append(len(probs))
        
        self.child_distributions = child_distributions
        self.num_children = torch.tensor(num_children)
        self.cum_num_children = torch.cumsum(self.num_children, dim=0)
        self.num_types = len(parent_probs)
        self.separate_children = separate_children

    
    def encode_child(self, parent, child):
        """
        Encodes a parent-child pair as a single integer.
        """
        if self.separate_children:
            return self.num_types + self.cum_num_children[parent] + child
        else:
            return self.num_types + child


    

    def generate(self, num_samples, horizon):
        """
        Generates a batch of size num_samples of GW data, each with a fixed horizon of length 2 * horizon.
        """

        data_set = torch.zeros(num_samples, 2 * horizon, dtype=torch.int32)
        for i in range(horizon):
            data_set[:, 2*i] = self.parent_distibution.sample((num_samples,))
            for j in range(num_samples):
                parent = data_set[j, 2*i]
                child = self.child_distributions[parent].sample()
                data_set[j, 2*i+1] = self.encode_child(parent, child)


        return data_set
    
    def _decode_data(self, parent_child_pairs):
        """
        Decodes a dataset into parent-child pairs.
        """
        if self.separate_children:
            return torch.stack([parent_child_pairs[:,0], parent_child_pairs[:,1] - self.cum_num_children[parent_child_pairs[:,0]] - self.num_types], dim=1)

        else:
            return torch.stack([parent_child_pairs[:,0], parent_child_pairs[:,1] - self.num_types], dim=1)

    def get_logits(self, dataset):
        """
        Returns the true logits of a dataset
        """
        horizon = dataset.shape[1] // 2
        logits = torch.zeros_like(dataset, dtype=torch.float32)
        logits[:,::2] = self.parent_distibution.logits[dataset[:,::2]]
        for i in range(horizon):
            parent_child_pairs = self._decode_data(dataset[:,2*i:2*i+2])
            for j in range(dataset.shape[0]):
                parent, child = parent_child_pairs[j]
                logits[j,2*i+1] = self.child_distributions[parent].logits[child]
        return logits.sum(dim=1)