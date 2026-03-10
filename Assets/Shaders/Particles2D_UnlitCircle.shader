Shader "Custom/Particles2D_UnlitCircle"
{
    Properties
    {
        _Color       ("Color", Color)                  = (0, 1, 1, 1)
        _EdgeSoftness("Edge Softness", Range(0, 0.5))  = 0.05
    }

    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex   vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv     : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv  : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            UNITY_INSTANCING_BUFFER_START(Props)
                UNITY_DEFINE_INSTANCED_PROP(float4, _Color)
            UNITY_INSTANCING_BUFFER_END(Props)

            float _EdgeSoftness;

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv  = v.uv;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(i);
                float2 uv    = i.uv * 2.0 - 1.0;
                float  dist  = length(uv);
                float  alpha = 1.0 - smoothstep(1.0 - _EdgeSoftness * 2.0, 1.0, dist);
                if (alpha < 0.001) discard;
                float4 col = UNITY_ACCESS_INSTANCED_PROP(Props, _Color);
                col.a *= alpha;
                return col;
            }
            ENDCG
        }
    }
}
