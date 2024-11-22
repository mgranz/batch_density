def batch_density(
    samples: torch.Tensor,
    n_bins: int = 100,
    density: torch.Tensor = None,
    bins: torch.Tensor = None,
    use_quantiles: bool = False,
    calculate_sample_density: bool = True,
    device: str = "cpu",
) -> Union[torch.Tensor, torch.Tensor]:
    """Generates a histogram (density, bins) for multiple dimensions for batch tensor shaped [n_samples, n_dims].

    Args:
        samples (torch.Tensor): Input samples.
        n_bins (int, optional): Number of bins. Defaults to 100.
        density (torch.Tensor, optional): Density tensor, if given use this density to calculate the density of the samples. Defaults to None.
        bins (torch.Tensor, optional): Bin edges, if given the bins will not be calculated. Defaults to None.
        use_quantiles (bool, optional): If True, uses 0.01 and 0.99 quantiles to calculate the bins. Defaults to False.
        calculate_sample_density (bool, optional): If True, calculates the density of each sample. Defaults to True.
        device (str, optional): Cuda device or CPU. Defaults to "cpu".

    Returns:
        Union[torch.Tensor, torch.Tensor]: sample_densities, density, bins (shaped [n_bins, n_dims] or [n_bins + 1, n_dims])
    """
    samples = samples.to(device)
    if bins is None:
        if use_quantiles:
            min = samples.quantile(0.01, dim=0)
            max = samples.quantile(0.99, dim=0)
        else:
            min = samples.min(dim=0)[0]
            max = samples.max(dim=0)[0]
        bins = linspace(min, max + 1e-5, num=n_bins + 1)
        bins = torch.cat(
            [bins, torch.ones(1, bins.shape[-1]).to(bins.device) * np.inf], dim=0
        )
        bins = torch.cat(
            [-torch.ones(1, bins.shape[-1]).to(bins.device) * np.inf, bins], dim=0
        )
    bins = bins.to(device)
    bin_length = bins[2:3] - bins[1:2] + 1e-12

    ids = torch.logical_and(
        (samples[None] < bins[1:, None]), (samples[None] >= bins[:-1, None])
    )
    ids = ids.to(torch.float32)

    if density is None:
        density = ids.mean(dim=1)
        density /= bin_length.to(device)

        density_q = None
    else:
        density.to(device)
        density_q /= bin_length.to(density_q.device)

    if calculate_sample_density:
        sample_densities = calculate_and_sample_densities(
            density,
            ids,
        )

    else:
        sample_densities = None

    if density_q is not None:
        return sample_densities, density_q, bins
    return sample_densities, density, bins
