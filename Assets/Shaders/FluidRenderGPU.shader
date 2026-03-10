Shader "Custom/FluidRenderGPU"
{
    Properties
    {
        _ParticleSize ("Particle Size", Float) = 0.08
        _ColorA ("Color Fast", Color) = (1, 0.3, 0.05, 1)
        _ColorB ("Color Slow", Color) = (0.05, 0.35, 1, 1)
        _SpeedScale ("Speed Scale (for color)", Float) = 8.0
    }

    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" "IgnoreProjector" = "True" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex   vert
            #pragma fragment frag
            #pragma target   4.5
            #pragma multi_compile_instancing
            #pragma instancing_options procedural:ConfigureProcedural

            #include "UnityCG.cginc"

            // GPU particle data: (pos.x, pos.y, speedSq, unused)
            StructuredBuffer<float4> _RenderData;

            float _ParticleSize;
            float4 _ColorA;
            float4 _ColorB;
            float  _SpeedScale;

            // Called once per instance before vert(); sets unity_ObjectToWorld
            void ConfigureProcedural()
            {
            #if defined(UNITY_PROCEDURAL_INSTANCING_ENABLED)
                float4 d = _RenderData[unity_InstanceID];
                // Scale by _ParticleSize, translate to world position
                unity_ObjectToWorld  = 0;
                unity_ObjectToWorld._m00 = _ParticleSize;
                unity_ObjectToWorld._m11 = _ParticleSize;
                unity_ObjectToWorld._m22 = _ParticleSize;
                unity_ObjectToWorld._m03 = d.x;
                unity_ObjectToWorld._m13 = d.y;
                unity_ObjectToWorld._m33 = 1;
                unity_WorldToObject  = 0;
                unity_WorldToObject._m00 = 1.0 / _ParticleSize;
                unity_WorldToObject._m11 = 1.0 / _ParticleSize;
                unity_WorldToObject._m22 = 1.0 / _ParticleSize;
                unity_WorldToObject._m03 = -d.x / _ParticleSize;
                unity_WorldToObject._m13 = -d.y / _ParticleSize;
                unity_WorldToObject._m33 = 1;
            #endif
            }

            struct Appdata
            {
                float4 vertex : POSITION;
                float2 uv     : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct V2F
            {
                float4 pos   : SV_POSITION;
                float2 uv    : TEXCOORD0;
                float4 color : COLOR;
            };

            V2F vert(Appdata v)
            {
                UNITY_SETUP_INSTANCE_ID(v);

                V2F o;
                o.pos = UnityObjectToClipPos(v.vertex);
                // Remap UV [0,1] -> [-1,1] for distance check
                o.uv = v.uv * 2.0 - 1.0;

            #if defined(UNITY_PROCEDURAL_INSTANCING_ENABLED)
                float speedSq = _RenderData[unity_InstanceID].z;
                float t = saturate(speedSq / (_SpeedScale * _SpeedScale));
                o.color = lerp(_ColorB, _ColorA, t);
            #else
                o.color = _ColorB;
            #endif
                return o;
            }

            fixed4 frag(V2F i) : SV_Target
            {
                float d = length(i.uv);
                // Soft circle: full alpha inside 0.7, fade to 0 at 1.0
                float alpha = 1.0 - smoothstep(0.65, 1.0, d);
                clip(alpha - 0.01);
                return float4(i.color.rgb, alpha);
            }
            ENDCG
        }
    }
}
