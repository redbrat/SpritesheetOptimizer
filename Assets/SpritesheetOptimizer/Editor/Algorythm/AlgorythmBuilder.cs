using System;
using System.Collections.Generic;

public class AlgorythmBuilder
{
    private IList<Type> _sizingCofiguratorTypes = new List<Type>();
    private Type _areaEnumeratorType;

    public AlgorythmBuilder()
    {
    }

    public AlgorythmBuilder SetAreaEnumerator<T>() where T : IAreaEnumerator
    {
        _areaEnumeratorType = typeof(T);
        return this;
    }

    public AlgorythmBuilder AddSizingsConfigurator<T>() where T : ISizingsConfigurator
    {
        _sizingCofiguratorTypes.Add(typeof(T));
        return this;
    }

    public Algorythm Build(MyColor[][][] data)
    {
        var sizingCofigurators = new List<ISizingsConfigurator>();
        for (int i = 0; i < _sizingCofiguratorTypes.Count; i++)
        {
            var instance = Activator.CreateInstance(_sizingCofiguratorTypes[i]);
            sizingCofigurators.Add((ISizingsConfigurator)instance);
        }

        if (_areaEnumeratorType == null)
            _areaEnumeratorType = typeof(DefaultAreaEnumerator);

        return new Algorythm(data, _areaEnumeratorType, sizingCofigurators);
    }
}
