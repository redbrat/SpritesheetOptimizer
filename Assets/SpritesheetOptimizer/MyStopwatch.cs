using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

public class MyStopwatch
{
    private Dictionary<string, Stopwatch> _sws = new Dictionary<string, Stopwatch>();

    public void Start(string id)
    {
        var stopWatch = getStopWatch(id);
        stopWatch.Start();
    }

    public void Stop(string id)
    {
        var stopWatch = getStopWatch(id);
        stopWatch.Stop();
    }

    private Stopwatch getStopWatch(string id)
    {
        if (!_sws.ContainsKey(id))
            _sws.Add(id, new Stopwatch());
        return _sws[id];
    }

    public string PrintResults()
    {
        var sb = new StringBuilder();

        foreach (var kvp in _sws)
        {
            sb.AppendLine($"{kvp.Key}: {kvp.Value.Elapsed}");
        }

        return sb.ToString();
    }
}
