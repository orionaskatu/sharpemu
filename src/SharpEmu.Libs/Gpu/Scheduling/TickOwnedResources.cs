// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Scheduling;

// A GPU feedback resource belongs to one submission until its completion callback
// has consumed it. Later submissions must never reuse its mapped storage.
// With a recycler, a consumed resource is reset and kept for the next submission instead of being
// destroyed: creating and freeing device memory on every submission cost more than the draws.
internal sealed class TickOwnedResources<T> : IDisposable where T : class, IDisposable
{
    private const int MaxPooled = 8;

    private readonly Dictionary<ulong, T> _resources = [];
    private readonly Stack<T> _pool = new();
    private readonly Action<T>? _recycle;
    private readonly object _gate = new();

    public TickOwnedResources(Action<T>? recycle = null) => _recycle = recycle;

    public T Acquire(ulong tick, Func<T> create)
    {
        lock (_gate)
        {
            if (!_resources.TryGetValue(tick, out var resource))
            {
                resource = _pool.Count != 0 ? _pool.Pop() : create();
                _resources.Add(tick, resource);
            }
            return resource;
        }
    }

    public void Complete(ulong tick, Action<T> consume)
    {
        lock (_gate)
        {
            if (!_resources.Remove(tick, out var resource)) return;
            var recycled = false;
            try
            {
                consume(resource);
                if (_recycle is not null && _pool.Count < MaxPooled)
                {
                    _recycle(resource);
                    _pool.Push(resource);
                    recycled = true;
                }
            }
            finally
            {
                if (!recycled) resource.Dispose();
            }
        }
    }

    // The caller must wait for GPU completion before disposing pending resources.
    public void Dispose()
    {
        lock (_gate)
        {
            foreach (var resource in _resources.Values) resource.Dispose();
            _resources.Clear();
            foreach (var resource in _pool) resource.Dispose();
            _pool.Clear();
        }
    }
}
