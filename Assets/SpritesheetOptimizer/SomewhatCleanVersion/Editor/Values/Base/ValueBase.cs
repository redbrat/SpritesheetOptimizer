using UnityEngine;

public abstract class ValueBase<T> : ScriptableObject
{
    public T Value => _value;

    [SerializeField]
    private T _value;
}
