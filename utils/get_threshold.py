def get_threshold(model, target_sparsity, get_sparsity, head=0, tail=1, margin=0.001):
    """
    get a appropriate threshold to obtain specified model sparsity
    model: the undergoing model
    target_sparsity: the target sparsity
    get_sparsity: a function that computes the sparsity given a model and a threshold
    head, tail: the starting and ending point of a search
    margin: the margin below which the sparsity will be accepted
    """
    head_s, tail_s = get_sparsity(model, thres=head), get_sparsity(model, thres=tail)
    
    sparsity = get_sparsity(model, thres=float(head+tail)/2)
    
    if abs(sparsity - target_sparsity) <= margin:
        # the ONLY output port
        return float(head + tail) / 2
    
    else:
        if get_sparsity(model, float(head+tail)/2) >= target_sparsity:
            return get_threshold(model, target_sparsity, get_sparsity, head=head, tail=float(head+tail)/2)
        else:
            return get_threshold(model, target_sparsity, get_sparsity, head=float(head+tail)/2, tail=tail)
        
