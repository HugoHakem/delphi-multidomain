import torch.profiler as profiler

def profile_and_print(step_fn, dataloader, optimizer, n_steps=2, top_k=20):
    """
    Runs a short profiling session and prints the top CUDA-heavy ops
    with their call stacks (filtered to skip PyTorch internals).
    """
    with profiler.profile(
      activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
      record_shapes=True,
      with_stack=True,
      profile_memory=True,
      with_flops=True,
      experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        for step, batch in enumerate(dataloader):
            step_fn(batch, step, 0)
            prof.step()
            if step >= n_steps:
                break

    #events = prof.key_averages()
    # events = sorted(events, key=lambda e: getattr(e, "cuda_time_total", 0), reverse=True)

    events = [e for e in prof.events() if hasattr(e, "cuda_time_total")]
    events = sorted(events, key=lambda e: getattr(e, "cuda_time_total", 0), reverse=True)

    print(f"\nTop {top_k} individual CUDA events with call sites:\n")
    for evt in events[:top_k]:
        print(f"=== {evt.name} | CUDA time: {evt.cuda_time/1e3:.2f} ms ===")
        try:
            for frame in evt.stack():
                fname, line, fn = frame
                if "site-packages" in fname:
                    continue  # skip library internals
                print(f"  {fname}:{line} in {fn}")
        except Exception:
            pass
        print()
    return
    print(f"\nTop {top_k} CUDA ops with stack traces:\n")
    for evt in events[:top_k]:
        print(f"=== {evt.key} | CUDA total: {getattr(evt, 'cuda_time_total', 0)/1e3:.2f} ms ===")
        try:
            for frame in evt.stack():
                fname, line, fn = frame
                if "site-packages" in fname:
                    continue  # skip library internals
                print(f"  {fname}:{line} in {fn}")
        except Exception:
            continue
        print()  # blank line between ops