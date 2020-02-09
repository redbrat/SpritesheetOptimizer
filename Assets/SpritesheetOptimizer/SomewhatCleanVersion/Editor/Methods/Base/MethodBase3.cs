using UnityEngine;

public abstract class MethodBase3<T1, T2, T3> : ScriptableObject
{
    public abstract void Invoke(T1 param1, T2 param2, T3 param3);
}
