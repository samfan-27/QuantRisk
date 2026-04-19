def approx_greek_fd(bump_func, base_val, h, is_gamma=False, richardson=True):
    """
    Generalized finite-difference calculator with Richardson Extrapolation.
    
    Parameters:
        bump_func (callable): A lambda function that takes a bump step (h) and returns the bumped option value.
        base_val (float): The unbumped base value of the option.
        h (float): The bump size.
        is_gamma (bool): If True, uses the second-order central difference formula.
        richardson (bool): If True, applies Richardson Extrapolation to stabilize the Greek.
        
    Returns:
        float: The approximated Greek value.
    """
    def diff(h_step):
        val_up = bump_func(h_step)
        val_dn = bump_func(-h_step)
        if is_gamma:
            return (val_up - 2 * base_val + val_dn) / (h_step ** 2)
        return (val_up - val_dn) / (2 * h_step)
    
    if richardson:
        d_h = diff(h)
        d_h2 = diff(h / 2.0)
        return (4 * d_h2 - d_h) / 3.0
    
    return diff(h)
