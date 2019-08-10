using System;
using System.Collections.Generic;

public class AlgorythmBuilder
{
    private IList<Type> _scoreCounterTypes = new List<Type>();
    private IList<Type> _sizingCofiguratorTypes = new List<Type>();
    private IList<object[]> _sizingCofiguratorCtorParams = new List<object[]>();
    private Type _areaEnumeratorType;
    private int _areasFreshmentSpan;
    private int _areasVolatilityRange;

    public AlgorythmBuilder()
    {
    }

    public AlgorythmBuilder SetAreaEnumerator<T>() where T : IAreaEnumerator
    {
        _areaEnumeratorType = typeof(T);
        return this;
    }

    public AlgorythmBuilder AddSizingsConfigurator<T>(params object[] parameters) where T : ISizingsConfigurator
    {
        _sizingCofiguratorTypes.Add(typeof(T));
        _sizingCofiguratorCtorParams.Add(parameters);
        return this;
    }

    public AlgorythmBuilder AddScoreCounter<T>() where T : IScoreCounter
    {
        _scoreCounterTypes.Add(typeof(T));
        return this;
    }

    public AlgorythmBuilder SetAreasFreshmentSpan(int span)
    {
        _areasFreshmentSpan = span;
        return this;
    }

    public AlgorythmBuilder SetAreasVolatilityRange(int range)
    {
        _areasVolatilityRange = range;
        return this;
    }

    public Algorythm Build(MyColor[][][] data)
    {
        var sizingCofigurators = new List<ISizingsConfigurator>();
        for (int i = 0; i < _sizingCofiguratorTypes.Count; i++)
        {
            var instance = Activator.CreateInstance(_sizingCofiguratorTypes[i], _sizingCofiguratorCtorParams[i]);
            sizingCofigurators.Add((ISizingsConfigurator)instance);
        }

        var scoreCounters = new List<IScoreCounter>();
        for (int i = 0; i < _scoreCounterTypes.Count; i++)
        {
            var instance = Activator.CreateInstance(_scoreCounterTypes[i]);
            scoreCounters.Add((IScoreCounter)instance);
        }

        if (_areaEnumeratorType == null)
            _areaEnumeratorType = typeof(DefaultAreaEnumerator);

        return new Algorythm(data, _areaEnumeratorType, sizingCofigurators, scoreCounters, _areasFreshmentSpan, _areasVolatilityRange);
    }
}
