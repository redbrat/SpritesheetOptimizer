using UnityEngine;

public abstract class MethodBase1<T> : ScriptableObject
{
    public abstract void Invoke(T param);
}
