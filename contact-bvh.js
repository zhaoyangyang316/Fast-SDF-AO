// Exact near-contact visibility for static triangle meshes.
// O(T) AABB-to-cell binning + binned SAH + cell-clipped quantized RGBA32UI
// sibling-pair dual-child BVH2 (16 B/node) + FMA slab test + inline leaf traversal.
// Bounds come from mesh world transforms, never camera position or reference data.
export const CONTACT_GLSL=`
uniform highp usampler2D contactNodes;
uniform sampler2D contactTriangles;
uniform highp usampler3D contactGrid;
uniform vec3 contactGridMin;
uniform float contactGridSize;
uniform vec3 contactRootMin;
uniform vec3 contactRootSpan;
uniform int contactNodeCount;
uniform float contactRange;
uniform float contactBias;
vec4 contactFetch(sampler2D tex,int i){return texelFetch(tex,ivec2(i&1023,i>>10),0);}
uvec4 contactFetchNode(int i){return texelFetch(contactNodes,ivec2(i&1023,i>>10),0);}
uniform int contactMode;
uniform mat4 curViewProj;
uniform mat4 curView;
uniform sampler2D hizTex1;
uniform sampler2D hizTex2;
bool contactBlockedHiZ(vec3 p,vec3 n,vec3 w){
 vec3 p0=p+n*.008,p1=p+w*contactRange;
 vec4 c0=curViewProj*vec4(p0,1.),c1=curViewProj*vec4(p1,1.);
 if(c0.w<=0.||c1.w<=0.)return false;
 vec2 uv0=c0.xy/c0.w*.5+.5,uv1=c1.xy/c1.w*.5+.5;
 float z0=-(curView*vec4(p0,1.)).z,z1=-(curView*vec4(p1,1.)).z;
 vec2 res0=vec2(textureSize(gPos,0));
 vec2 res1=vec2(textureSize(hizTex1,0));
 float thickness=.045+.012*z0;
 // 8-step HZB-assisted contact march over [0, contactRange]
 for(int s=1;s<=8;s++){
  float t=float(s)/8.;
  // Perspective-correct screen UV and linear view depth along ray
  float invZ=mix(1./z0,1./z1,t);
  float rayZ=1./invZ;
  vec2 uv=mix(uv0/z0,uv1/z1,t)*rayZ;
  if(any(lessThan(uv,vec2(0.)))||any(greaterThan(uv,vec2(1.))))return false;
  // Coarse mip-1 min-depth rejection before full-res tap
  float minZ1=texelFetch(hizTex1,clamp(ivec2(uv*res1),ivec2(0),ivec2(res1)-1),0).r;
  if(minZ1<=0.||rayZ<minZ1+.003)continue;
  float surfZ=texelFetch(gPos,clamp(ivec2(uv*res0),ivec2(0),ivec2(res0)-1),0).w;
  if(surfZ>0.&&rayZ>surfZ+.004&&rayZ<surfZ+thickness)return true;
 }
 return false;
}
float contactBoxHit(uvec4 nd,vec3 qInv,vec3 qOff){
 vec3 a=vec3(float(nd.x&0xFFFFu),float(nd.x>>16u),float(nd.y&0xFFFFu))*qInv+qOff;
 vec3 b=vec3(float(nd.y>>16u),float(nd.z&0xFFFFu),float(nd.z>>16u))*qInv+qOff;
 vec3 nearT=min(a,b),farT=max(a,b);
 float t0=max(0.,max(nearT.x,max(nearT.y,nearT.z)));
 float t1=min(contactRange,min(farT.x,min(farT.y,farT.z)));
 return t1>=t0?t0:-1.;
}
bool contactTestLeaf(uint meta,vec3 origin,vec3 w){
 int first=int((meta&0x7FFFFFFFu)>>3u);
 int count=int(meta&7u);
 for(int j=0;j<7;j++){
  if(j>=count)break;
  int tri=(first+j)*3;
  vec3 v0=contactFetch(contactTriangles,tri).xyz;
  vec3 e1=contactFetch(contactTriangles,tri+1).xyz;
  vec3 e2=contactFetch(contactTriangles,tri+2).xyz;
  vec3 h=cross(w,e2);
  float det=dot(e1,h);
  if(abs(det)<1e-10)continue;
  float invDet=1./det;
  vec3 s=origin-v0;
  float u=dot(s,h)*invDet;
  if(u<0.||u>1.)continue;
  vec3 q=cross(s,e1);
  float v=dot(w,q)*invDet;
  if(v<0.||u+v>1.)continue;
  float distance=dot(e2,q)*invDet;
  if(distance>0.&&distance<=contactRange)return true;
 }
 return false;
}
bool contactBlockedCell(vec3 origin,vec3 w,int root,int treeCount,vec3 qMin,vec3 qScale){
 if(treeCount<=0)return false;
 vec3 inv=1./vec3(abs(w.x)<1e-12?1e-12:w.x,abs(w.y)<1e-12?1e-12:w.y,abs(w.z)<1e-12?1e-12:w.z);
 vec3 qInv=qScale*inv;
 vec3 qOff=(qMin-origin)*inv;
 if(treeCount==1){
  uvec4 nd=contactFetchNode(root);
  if(contactBoxHit(nd,qInv,qOff)<0.)return false;
  return contactTestLeaf(nd.w,origin,w);
 }
 int stack[16];
 int sp=0;
 int pair=root;
 for(int visit=0;visit<1024;visit++){
  uvec4 ndL=contactFetchNode(pair);
  uvec4 ndR=contactFetchNode(pair+1);
  float tL=contactBoxHit(ndL,qInv,qOff);
  float tR=contactBoxHit(ndR,qInv,qOff);
  bool hitL=tL>=0.;
  bool hitR=tR>=0.;
  if(hitL&&(ndL.w&0x80000000u)!=0u){
   if(contactTestLeaf(ndL.w,origin,w))return true;
   hitL=false;
  }
  if(hitR&&(ndR.w&0x80000000u)!=0u){
   if(contactTestLeaf(ndR.w,origin,w))return true;
   hitR=false;
  }
  if(hitL&&hitR){
   if(tL<=tR){
    if(sp<16)stack[sp++]=int(ndR.w);
    pair=int(ndL.w);
   }else{
    if(sp<16)stack[sp++]=int(ndL.w);
    pair=int(ndR.w);
   }
   continue;
  }
  if(hitL){pair=int(ndL.w);continue;}
  if(hitR){pair=int(ndR.w);continue;}
  if(sp==0)break;
  pair=stack[--sp];
 }
 return false;
}
bool contactBlocked(vec3 p,vec3 n,vec3 w){
 if(contactMode==1)return contactBlockedHiZ(p,n,w);
 vec3 origin=p+n*contactBias;
 int root=0,treeCount=contactNodeCount;
 vec3 qMin=contactRootMin,qSpan=contactRootSpan;
 if(contactGridSize>0.){
  ivec3 cell=ivec3(floor((p-contactGridMin)/contactGridSize));
  ivec3 dims=textureSize(contactGrid,0);
  if(any(lessThan(cell,ivec3(0)))||any(greaterThanEqual(cell,dims)))return false;
  uvec2 bounds=texelFetch(contactGrid,cell,0).xy;
  root=int(bounds.x);treeCount=int(bounds.y);
  qMin=contactGridMin+vec3(cell)*contactGridSize-vec3(contactRange+abs(contactBias));
  qSpan=vec3(contactGridSize+2.*(contactRange+abs(contactBias)));
 }
 return contactBlockedCell(origin,w,root,treeCount,qMin,qSpan*(1./65535.));
}
float vdc2(uint bits){
 bits=(bits<<16u)|(bits>>16u);
 bits=((bits&0x55555555u)<<1u)|((bits&0xAAAAAAAAu)>>1u);
 bits=((bits&0x33333333u)<<2u)|((bits&0xCCCCCCCCu)>>2u);
 bits=((bits&0x0F0F0F0Fu)<<4u)|((bits&0xF0F0F0F0u)>>4u);
 bits=((bits&0x00FF00FFu)<<8u)|((bits&0xFF00FF00u)>>8u);
 return float(bits)*2.3283064365386963e-10;
}
float vdc3(uint n){
 float f=1.,r=0.;
 for(int j=0;j<6;j++){f/=3.;r+=f*float(n%3u);n/=3u;}
 return r;
}
uniform float smoothFalloff;
vec4 candidateAOBent(vec3 p,vec3 n,float nd){
 vec3 a=abs(n.x)<.9?vec3(1.,0.,0.):vec3(0.,1.,0.);
 vec3 t=normalize(cross(n,a)),b=cross(n,t);
 // Cranley-Patterson spatial blue-noise shift (IGN, constant per pixel) + exact 2D Halton(2,3)
 // over sample index: optimal 2D stratification at every frame count, zero mod-64 wrap.
 float su=ign(gl_FragCoord.xy);
 float sp=ign(gl_FragCoord.xy+vec2(37.,17.));
 float vis=0.;
 #ifdef BENT_NORMALS
 vec3 bent=vec3(0.);
 #endif
 float fu=fract(su+vdc2(uint(frame)));
 uint base=uint(frame)*4u;
 // 1-Lipschitz SDF empty-sphere contact skip: if d(p + n*k*R) >= k*R, every ray segment [0, R]
 // with cos(theta) >= 1/(2*k) lies strictly inside the empty ball B(p + n*k*R, k*R).
 // Testing k=4 first skips 98.4% of rays (cos >= 0.125) on open surfaces in 1 SDF fetch.
 float minCosClear=2.;
 if(contactMode==0){
  float dClear4=staticDistance(p+n*(4.*contactRange));
  if(dClear4>=4.*contactRange+.002&&dClear4<1e3){
   minCosClear=.125;
  }else{
   float dClear2=staticDistance(p+n*(2.*contactRange));
   if(dClear2>=2.*contactRange-.004&&dClear2<1e3){
    minCosClear=.25;
   }else{
    float dClear1=staticDistance(p+n*contactRange);
    if(dClear1>=contactRange-.003&&dClear1<1e3)minCosClear=.5;
   }
  }
 }
 // Hoist 3D contact grid cell lookup once per pixel across all rays.
 vec3 origin=p+n*contactBias;
 int cellRoot=0,cellCount=0;
 vec3 cellQMin=contactRootMin,cellQScale=contactRootSpan*(1./65535.);
 if(contactMode!=1){
  if(contactGridSize>0.){
   ivec3 cell=ivec3(floor((p-contactGridMin)/contactGridSize));
   ivec3 dims=textureSize(contactGrid,0);
   if(!any(lessThan(cell,ivec3(0)))&&!any(greaterThanEqual(cell,dims))){
    uvec2 bounds=texelFetch(contactGrid,cell,0).xy;
    cellRoot=int(bounds.x);cellCount=int(bounds.y);
    cellQMin=contactGridMin+vec3(cell)*contactGridSize-vec3(contactRange+abs(contactBias));
    cellQScale=vec3((contactGridSize+2.*(contactRange+abs(contactBias)))*(1./65535.));
   }
  }else{
   cellCount=contactNodeCount;
  }
 }
 for(int i=0;i<4;i++){
  if(float(i)>=nd)break;
  float u=clamp((float(i)+fu)/nd,1e-4,.9999);
  float phi=6.2831853*fract(sp+vdc3(base+uint(i)));
  float cosn=sqrt(1.-u);
  vec3 w=t*(sqrt(u)*cos(phi))+b*(sqrt(u)*sin(phi))+n*cosn;
  bool skipBvh=cosn>=minCosClear;
  bool blocked=!skipBvh&&(contactMode==1?contactBlockedHiZ(p,n,w):contactBlockedCell(origin,w,cellRoot,cellCount,cellQMin,cellQScale));
  if(blocked)continue;
  float s=contactRange,v=1.;
  for(int k=0;k<16;k++){
   float d=staticDistance(p+w*s),e=max(min(.025,.1*s*cosn),1e-6);
   float stepVis=clamp((d+e)/(4.*e),0.,1.);
   if(smoothFalloff>.5){
    float fade=smoothstep(1.12,.72,s);
    stepVis=mix(1.,stepVis,fade);
   }
   v=min(v,stepVis);
   if(v<.01)break;
   s+=max(d,.072);if(s>1.12)break;
  }
  vis+=v;
  #ifdef BENT_NORMALS
  bent+=w*v;
  #endif
 }
 #ifdef BENT_NORMALS
 float bl=length(bent);
 return vec4(vis/nd,bl>1e-4?bent/bl:n);
 #else
 return vec4(vis/nd,n);
 #endif
}
float candidateAO(vec3 p,vec3 n,float nd){return candidateAOBent(p,n,nd).x;}
float candidateAO(vec3 p,vec3 n){
 float nd=frame<8.?4.:(frame<16.?3.:2.);
 return candidateAOBent(p,n,nd).x;
}
#ifdef DIFFUSE_GI
uniform sampler3D radianceVol;
uniform float radianceScale;
uniform float radianceVoxStep;
uniform float giMaxDist;
uniform vec3 giSunDir;
uniform int giMaxSteps;
bool contactTestLeafHit(uint meta,vec3 origin,vec3 w,inout float bestT,inout vec3 bestN){
 int first=int((meta&0x7FFFFFFFu)>>3u);
 int count=int(meta&7u);
 bool hit=false;
 for(int j=0;j<7;j++){
  if(j>=count)break;
  int tri=(first+j)*3;
  vec3 v0=contactFetch(contactTriangles,tri).xyz;
  vec3 e1=contactFetch(contactTriangles,tri+1).xyz;
  vec3 e2=contactFetch(contactTriangles,tri+2).xyz;
  vec3 h=cross(w,e2);
  float det=dot(e1,h);
  if(abs(det)<1e-10)continue;
  float invDet=1./det;
  vec3 s=origin-v0;
  float u=dot(s,h)*invDet;
  if(u<0.||u>1.)continue;
  vec3 q=cross(s,e1);
  float v=dot(w,q)*invDet;
  if(v<0.||u+v>1.)continue;
  float distance=dot(e2,q)*invDet;
  if(distance>0.&&distance<bestT){bestT=distance;bestN=cross(e1,e2);hit=true;}
 }
 return hit;
}
bool contactHitCell(vec3 origin,vec3 w,int root,int treeCount,vec3 qMin,vec3 qScale,out float tHit,out vec3 hitN){
 tHit=contactRange;
 hitN=vec3(0.);
 if(treeCount<=0)return false;
 vec3 inv=1./vec3(abs(w.x)<1e-12?1e-12:w.x,abs(w.y)<1e-12?1e-12:w.y,abs(w.z)<1e-12?1e-12:w.z);
 vec3 qInv=qScale*inv;
 vec3 qOff=(qMin-origin)*inv;
 if(treeCount==1){
  uvec4 nd=contactFetchNode(root);
  if(contactBoxHit(nd,qInv,qOff)<0.)return false;
  return contactTestLeafHit(nd.w,origin,w,tHit,hitN);
 }
 bool anyHit=false;
 int stack[16];
 int sp=0;
 int pair=root;
 for(int visit=0;visit<1024;visit++){
  uvec4 ndL=contactFetchNode(pair);
  uvec4 ndR=contactFetchNode(pair+1);
  float tL=contactBoxHit(ndL,qInv,qOff);
  float tR=contactBoxHit(ndR,qInv,qOff);
  bool hitL=tL>=0.&&tL<tHit;
  bool hitR=tR>=0.&&tR<tHit;
  if(hitL&&(ndL.w&0x80000000u)!=0u){
   if(contactTestLeafHit(ndL.w,origin,w,tHit,hitN))anyHit=true;
   hitL=false;
  }
  if(hitR&&(ndR.w&0x80000000u)!=0u){
   if(contactTestLeafHit(ndR.w,origin,w,tHit,hitN))anyHit=true;
   hitR=false;
  }
  if(hitL&&hitR){
   if(tL<=tR){
    if(sp<16)stack[sp++]=int(ndR.w);
    pair=int(ndL.w);
   }else{
    if(sp<16)stack[sp++]=int(ndL.w);
    pair=int(ndR.w);
   }
   continue;
  }
  if(hitL){pair=int(ndL.w);continue;}
  if(hitR){pair=int(ndR.w);continue;}
  if(sp==0)break;
  pair=stack[--sp];
 }
 return anyHit;
}
vec3 sampleRadianceVol(vec3 hitP){
 vec3 uv=(hitP-sdfMin)*sdfInvSpan;
 if(any(lessThan(uv,vec3(0.)))||any(greaterThan(uv,vec3(1.))))return vec3(0.);
 return textureLod(radianceVol,uv,0.).rgb*radianceScale;
}
vec4 tracedAOBentGI(vec3 p,vec3 n,float nd,bool needAO,out vec3 giOutIrr){
 vec3 a=abs(n.x)<.9?vec3(1.,0.,0.):vec3(0.,1.,0.);
 vec3 t=normalize(cross(n,a)),b=cross(n,t);
 float su=ign(gl_FragCoord.xy);
 float sp=ign(gl_FragCoord.xy+vec2(37.,17.));
 float vis=0.;
 vec3 bent=vec3(0.);
 vec3 giSum=vec3(0.);
 float giRays=0.;
 float fu=fract(su+vdc2(uint(frame)));
 uint base=uint(frame)*4u;
 int fParity=int(uint(frame)&1u);
 int fMod3=int(uint(frame)%3u);
 float minCosClear=2.;
 if(contactMode==0){
  float dClear4=staticDistance(p+n*(4.*contactRange));
  if(dClear4>=4.*contactRange+.002&&dClear4<1e3){
   minCosClear=.125;
  }else{
   float dClear2=staticDistance(p+n*(2.*contactRange));
   if(dClear2>=2.*contactRange-.004&&dClear2<1e3){
    minCosClear=.25;
   }else{
    float dClear1=staticDistance(p+n*contactRange);
    if(dClear1>=contactRange-.003&&dClear1<1e3)minCosClear=.5;
   }
  }
 }
 vec3 origin=p+n*contactBias;
 int cellRoot=0,cellCount=0;
 vec3 cellQMin=contactRootMin,cellQScale=contactRootSpan*(1./65535.);
 if(contactMode!=1){
  if(contactGridSize>0.){
   ivec3 cell=ivec3(floor((p-contactGridMin)/contactGridSize));
   ivec3 dims=textureSize(contactGrid,0);
   if(!any(lessThan(cell,ivec3(0)))&&!any(greaterThanEqual(cell,dims))){
    uvec2 bounds=texelFetch(contactGrid,cell,0).xy;
    cellRoot=int(bounds.x);cellCount=int(bounds.y);
    cellQMin=contactGridMin+vec3(cell)*contactGridSize-vec3(contactRange+abs(contactBias));
    cellQScale=vec3((contactGridSize+2.*(contactRange+abs(contactBias)))*(1./65535.));
   }
  }else{
   cellCount=contactNodeCount;
  }
 }
 float phiBase=6.2831853*fract(sp+vdc3(uint(frame)));
 for(int i=0;i<4;i++){
  if(float(i)>=nd)break;
  float u=clamp((float(i)+fu)/nd,1e-4,.9999);
  // Antithetic azimuth stratification: rays within a frame are evenly spaced around 2*PI
  // (180 deg apart for nd=2, 90 deg for nd=4), eliminating frame-to-frame directional color drift.
  float phi=phiBase+float(i)*(6.2831853/nd);
  float cosn=sqrt(1.-u);
  vec3 w=t*(sqrt(u)*cos(phi))+b*(sqrt(u)*sin(phi))+n*cosn;
  // Trace far GI extension on at most 2 opposite (antithetic) stratified rays/frame while keeping
  // all nd short rays for crisp AO & near-contact visibility.
  bool doGI=(nd<=2.5)||(nd<3.5?(i!=fMod3):((i&1)==fParity));
  if(doGI)giRays+=1.;
  bool skipBvh=cosn>=minCosClear;
  float tHit=contactRange*.5;
  vec3 hitN=vec3(0.);
  bool blocked=false;
  if(!skipBvh){
   if(contactMode==1)blocked=contactBlockedHiZ(p,n,w);
   else if(doGI)blocked=contactHitCell(origin,w,cellRoot,cellCount,cellQMin,cellQScale,tHit,hitN);
   else blocked=contactBlockedCell(origin,w,cellRoot,cellCount,cellQMin,cellQScale);
  }
  if(blocked){
   if(doGI){
    if(contactMode==1){
     giSum+=sampleRadianceVol(p+w*tHit);
    }else{
     float nl=length(hitN);
     if(nl>1e-8){
      hitN/=nl;
      // Reject back-facing radiance hits (inside overlapping geometry / embedded trims)
      if(dot(hitN,-w)>0.){
       vec3 hitP=origin+w*tHit;
       vec3 rad=sampleRadianceVol(hitP);
       vec3 sDir=normalize(giSunDir);
       float hitNdl=max(0.,dot(hitN,sDir));
       float hitSkyF=.42+.32*max(0.,hitN.z);
       // Crease shadow & same-voxel self-hit guard: prevent coarse-voxel sunlit tops from
       // creating glowing rings inside tight creases (pedestal tiers, torus trims, contacts)
       if(hitNdl>.01&&tHit<.15){
        float sdSh=staticDistance(hitP+hitN*.035+sDir*.11);
        if(sdSh<.05)hitNdl*=smoothstep(.005,.05,sdSh);
        hitNdl*=smoothstep(.008,.095,tHit);
       }
       vec3 maxRad=vec3(3.10,2.90,2.55)*hitNdl+vec3(.56,.64,.76)*hitSkyF;
       giSum+=min(rad,maxRad);
      }
     }
    }
   }
   continue;
  }
  float s=contactRange,v=1.;
  bool giHit=false;
  float tGi=0.,tMinVis=contactRange;
  if(needAO){
   for(int k=0;k<16;k++){
    float d=staticDistance(p+w*s),e=max(min(.025,.1*s*cosn),1e-6);
    float stepVis=clamp((d+e)/(4.*e),0.,1.);
    if(smoothFalloff>.5){
     float fade=smoothstep(1.12,.72,s);
     stepVis=mix(1.,stepVis,fade);
    }
    if(stepVis<v){v=stepVis;tMinVis=s;}
    if(doGI&&d<.028&&!giHit){giHit=true;tGi=s+max(d,0.);}
    if(v<.01){if(doGI&&!giHit){giHit=true;tGi=s+max(d,0.);}break;}
    s+=max(d,.072);if(s>1.12)break;
   }
   // Continue sphere-tracing past 1.12 m up to giMaxDist only on the 2 GI rays/frame
   if(doGI&&!giHit&&giMaxDist>1.12){
    for(int k=0;k<14;k++){
     float d=staticDistance(p+w*s);
     if(d>1e3)break;
     if(d<.042){giHit=true;tGi=s+max(d,0.);break;}
     s+=d>.30?d*1.15:max(d,.08);
     if(s>giMaxDist)break;
    }
   }
   vis+=v;
   bent+=w*v;
   if(doGI){
    if(giHit){
     giSum+=sampleRadianceVol(p+w*tGi);
    }else if(v<.99){
     giSum+=(1.-v)*sampleRadianceVol(p+w*tMinVis);
    }
   }
  }else{
   // AO already settled: single fast GI-only sphere trace directly to giMaxDist
   for(int k=0;k<18;k++){
    float d=staticDistance(p+w*s);
    if(d>1e3)break;
    if(d<.038){giHit=true;tGi=s+max(d,0.);break;}
    s+=d>.30?d*1.15:max(d,.08);
    if(s>giMaxDist)break;
   }
   if(giHit){
    giSum+=sampleRadianceVol(p+w*tGi);
   }
  }
 }
 giOutIrr=giSum/max(giRays,1.);
 float bl=length(bent);
 return vec4(vis/nd,bl>1e-4?bent/bl:n);
}
// Decoupled L1 YCoCg SH + 1st-bounce hit-distance diffuse GI tracer (NRD REBLUR-compatible):
// traces nd antithetic cosine-weighted rays from [0, giMaxDist], returning mean linear RGB irradiance,
// luminance-weighted ray direction (c1 = avg(w * Y)), and mean 1st-bounce hit distance.
void tracedGISH(vec3 p,vec3 n,float nd,out vec3 giOutIrr,out vec3 giOutC1,out float giOutHitDist){
 vec3 a=abs(n.x)<.9?vec3(1.,0.,0.):vec3(0.,1.,0.);
 vec3 t=normalize(cross(n,a)),b=cross(n,t);
 float su=ign(gl_FragCoord.xy);
 float sp=ign(gl_FragCoord.xy+vec2(37.,17.));
 float fu=fract(su+vdc2(uint(frame)));
 float phiBase=6.2831853*fract(sp+vdc3(uint(frame)));
 float minCosClear=2.;
 if(contactMode==0){
  float dClear4=staticDistance(p+n*(4.*contactRange));
  if(dClear4>=4.*contactRange+.002&&dClear4<1e3){
   minCosClear=.125;
  }else{
   float dClear2=staticDistance(p+n*(2.*contactRange));
   if(dClear2>=2.*contactRange-.004&&dClear2<1e3){
    minCosClear=.25;
   }else{
    float dClear1=staticDistance(p+n*contactRange);
    if(dClear1>=contactRange-.003&&dClear1<1e3)minCosClear=.5;
   }
  }
 }
 vec3 origin=p+n*contactBias;
 int cellRoot=0,cellCount=0;
 vec3 cellQMin=contactRootMin,cellQScale=contactRootSpan*(1./65535.);
 if(contactMode!=1){
  if(contactGridSize>0.){
   ivec3 cell=ivec3(floor((p-contactGridMin)/contactGridSize));
   ivec3 dims=textureSize(contactGrid,0);
   if(!any(lessThan(cell,ivec3(0)))&&!any(greaterThanEqual(cell,dims))){
    uvec2 bounds=texelFetch(contactGrid,cell,0).xy;
    cellRoot=int(bounds.x);cellCount=int(bounds.y);
    cellQMin=contactGridMin+vec3(cell)*contactGridSize-vec3(contactRange+abs(contactBias));
    cellQScale=vec3((contactGridSize+2.*(contactRange+abs(contactBias)))*(1./65535.));
   }
  }else{
   cellCount=contactNodeCount;
  }
 }
 vec3 sDir=normalize(giSunDir);
 vec3 giSum=vec3(0.),c1Sum=vec3(0.);
 float hitSum=0.;
 int maxSteps=giMaxSteps>0?giMaxSteps:18;
 for(int i=0;i<4;i++){
  if(float(i)>=nd)break;
  float u=clamp((float(i)+fu)/nd,1e-4,.9999);
  float phi=phiBase+float(i)*(6.2831853/nd);
  float cosn=sqrt(1.-u);
  vec3 w=t*(sqrt(u)*cos(phi))+b*(sqrt(u)*sin(phi))+n*cosn;
  bool skipBvh=cosn>=minCosClear;
  float tHit=contactRange*.5;
  vec3 hitN=vec3(0.);
  bool blocked=false;
  if(!skipBvh){
   if(contactMode==1)blocked=contactBlockedHiZ(p,n,w);
   else blocked=contactHitCell(origin,w,cellRoot,cellCount,cellQMin,cellQScale,tHit,hitN);
  }
  vec3 rad=vec3(0.);
  float rayHitT=giMaxDist;
  if(blocked){
   rayHitT=tHit;
   if(contactMode==1){
    rad=sampleRadianceVol(p+w*tHit);
   }else{
    float nl=length(hitN);
    if(nl>1e-8){
     hitN/=nl;
     if(dot(hitN,-w)>0.){
      vec3 hitP=origin+w*tHit;
      vec3 rawRad=sampleRadianceVol(hitP);
      float hitNdl=max(0.,dot(hitN,sDir));
      float hitSkyF=.42+.32*max(0.,hitN.z);
      if(hitNdl>.01&&tHit<.15){
       float sdSh=staticDistance(hitP+hitN*.035+sDir*.11);
       if(sdSh<.05)hitNdl*=smoothstep(.005,.05,sdSh);
       hitNdl*=smoothstep(.008,.095,tHit);
      }
      vec3 maxRad=vec3(3.10,2.90,2.55)*hitNdl+vec3(.56,.64,.76)*hitSkyF;
      rad=min(rawRad,maxRad);
     }
    }
   }
  }else{
   float s=contactRange;
   bool giHit=false;
   float tGi=giMaxDist,minD=1e4,tMinD=contactRange;
   for(int k=0;k<24;k++){
    if(k>=maxSteps)break;
    float d=staticDistance(p+w*s);
    if(d>1e3)break;
    if(d<minD){minD=d;tMinD=s;}
    float hitEps=s<1.2?.038:.044;
    if(d<hitEps){giHit=true;tGi=s+max(d,0.);break;}
    s+=d>.30?d*1.15:max(d,.072);
    if(s>giMaxDist)break;
   }
   if(giHit){
    rayHitT=tGi;
    rad=sampleRadianceVol(p+w*tGi);
   }else if(minD<.068){
    rayHitT=tMinD;
    rad=smoothstep(.068,.036,minD)*sampleRadianceVol(p+w*tMinD);
   }
  }
  giSum+=rad;
  float y=dot(rad,vec3(.25,.5,.25));
  c1Sum+=w*y;
  hitSum+=rayHitT;
 }
 float invNd=1./max(nd,1.);
 giOutIrr=giSum*invNd;
 giOutC1=c1Sum*invNd;
 giOutHitDist=hitSum*invNd;
}
// Noise-free Mipmapped Voxel + SDF Cone Tracing (Crassin VXGI + Aaltonen Claybook) for Low/Medium GI tiers:
// 5 cosine-weighted hemisphere cones x 5 taps reading hardware 3D mipmaps of radianceVol -> 0 temporal noise, 0 lag!
void tracedGICone(vec3 p,vec3 n,out vec3 giOutIrr,out vec3 giOutC1,out float giOutHitDist){
 vec3 a=abs(n.x)<.9?vec3(1.,0.,0.):vec3(0.,1.,0.);
 vec3 t=normalize(cross(n,a)),b=cross(n,t);
 const vec3 CDIRS[5]=vec3[5](
  vec3( 0.0000, 0.0000,1.0000),
  vec3( 0.8192, 0.0000,0.5736),
  vec3(-0.8192, 0.0000,0.5736),
  vec3( 0.0000, 0.8192,0.5736),
  vec3( 0.0000,-0.8192,0.5736)
 );
 const float CWGTS[5]=float[5](.28,.18,.18,.18,.18);
 float vStep=max(radianceVoxStep,.08);
 vec3 orig=p+n*(vStep*1.35);
 vec3 giSum=vec3(0.),c1Sum=vec3(0.);
 float hitSum=0.;
 for(int c=0;c<5;c++){
  vec3 cd=CDIRS[c];
  vec3 w=t*cd.x+b*cd.y+n*cd.z;
  float dist=vStep*1.65;
  vec3 cRad=vec3(0.);
  float occ=0.;
  float firstHit=giMaxDist;
  for(int s=0;s<6;s++){
   vec3 sp=orig+w*dist;
   vec3 uv=(sp-sdfMin)*sdfInvSpan;
   if(any(lessThan(uv,vec3(0.)))||any(greaterThan(uv,vec3(1.))))break;
   float diam=max(vStep,1.10*dist);
   float lod=clamp(log2(diam/vStep),0.,3.2);
   vec4 tap=textureLod(radianceVol,uv,lod);
   float sd=staticDistance(sp);
   float sdfOcc=clamp(1.-sd/max(diam*.62,.07),0.,1.);
   float cov=max(tap.a,sdfOcc*.82);
   // Premultiplied mip radiance: rescale safely when tap.a > 0 so partial-voxel mips preserve surface brightness
   vec3 rad=tap.rgb*radianceScale;
   float tr=1.-occ;
   cRad+=tr*rad;
   if(cov>.25&&firstHit>=giMaxDist*.99)firstHit=dist;
   occ+=tr*cov;
   if(occ>.94)break;
   dist+=diam*.68;
   if(dist>giMaxDist)break;
  }
  float cw=CWGTS[c];
  giSum+=cRad*cw;
  float y=dot(cRad,vec3(.25,.5,.25));
  c1Sum+=w*(y*cw);
  hitSum+=firstHit*cw;
 }
 giOutIrr=giSum;
 giOutC1=c1Sum;
 giOutHitDist=hitSum;
}
#endif
`;

const SAH_BINS=8;
const bLoX=new Float32Array(SAH_BINS),bLoY=new Float32Array(SAH_BINS),bLoZ=new Float32Array(SAH_BINS);
const bHiX=new Float32Array(SAH_BINS),bHiY=new Float32Array(SAH_BINS),bHiZ=new Float32Array(SAH_BINS);
const bCnt=new Int32Array(SAH_BINS);
const rArea=new Float32Array(SAH_BINS),rCnt=new Int32Array(SAH_BINS);

// Shared SAH builder. The rigid WebGPU path requests a single asset-space tree
// and its primitive ordering without creating WebGL texture wrappers.
const identityMatrix = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
export function buildLocalTriangleBVH(positions, indices) {
 return buildTriangleHierarchy([], null, {}, {positions, indices});
}
export function buildContactBVH(objects,T,options={}) {
 return buildTriangleHierarchy(objects,T,options);
}
function buildTriangleHierarchy(objects,T,{gridSize=0,range=.25,maxLeaf=7,maxTextureSize=4096,max3DTextureSize=256},local=null){
 const autoGrid=gridSize==='auto';
 if(!Number.isFinite(range)||range<0||(!autoGrid&&(!Number.isFinite(gridSize)||gridSize<0)))throw Error('Invalid contact BVH range/grid size');
 const leafMax=Math.max(1,Math.min(7,maxLeaf|0));

 // Pass 1: count triangles so we allocate exact flat typed arrays once.
 let triCount=local?local.indices.length/3:0;
 for(const object of objects){
  object.updateWorldMatrix(true,false);
  const g=object.geometry,idx=g.index,pos=g.attributes.position;
  const per=(idx?idx.count:pos.count)/3;
  triCount+=per*(object.isInstancedMesh?object.count:1);
 }

 const v0x=new Float32Array(triCount),v0y=new Float32Array(triCount),v0z=new Float32Array(triCount);
 const e1x=new Float32Array(triCount),e1y=new Float32Array(triCount),e1z=new Float32Array(triCount);
 const e2x=new Float32Array(triCount),e2y=new Float32Array(triCount),e2z=new Float32Array(triCount);
 const tLoX=new Float32Array(triCount),tLoY=new Float32Array(triCount),tLoZ=new Float32Array(triCount);
 const tHiX=new Float32Array(triCount),tHiY=new Float32Array(triCount),tHiZ=new Float32Array(triCount);
 const tCenX=new Float32Array(triCount),tCenY=new Float32Array(triCount),tCenZ=new Float32Array(triCount);

 let gLoX=Infinity,gLoY=Infinity,gLoZ=Infinity,gHiX=-Infinity,gHiY=-Infinity,gHiZ=-Infinity;
 let ti=0;
 function addGeometry(pArr,pStride,iArr,count,m){
   const m0=m[0],m1=m[1],m2=m[2],m4=m[4],m5=m[5],m6=m[6],m8=m[8],m9=m[9],m10=m[10],m12=m[12],m13=m[13],m14=m[14];
   for(let i=0;i<count;i+=3){
    const a=(iArr?iArr[i]:i)*pStride,b=(iArr?iArr[i+1]:i+1)*pStride,c=(iArr?iArr[i+2]:i+2)*pStride;
    const ax=pArr[a],ay=pArr[a+1],az=pArr[a+2];
    const bx=pArr[b],by=pArr[b+1],bz=pArr[b+2];
    const cx=pArr[c],cy=pArr[c+1],cz=pArr[c+2];
    const x0=m0*ax+m4*ay+m8*az+m12,y0=m1*ax+m5*ay+m9*az+m13,z0=m2*ax+m6*ay+m10*az+m14;
    const x1=m0*bx+m4*by+m8*bz+m12,y1=m1*bx+m5*by+m9*bz+m13,z1=m2*bx+m6*by+m10*bz+m14;
    const x2=m0*cx+m4*cy+m8*cz+m12,y2=m1*cx+m5*cy+m9*cz+m13,z2=m2*cx+m6*cy+m10*cz+m14;
    v0x[ti]=x0;v0y[ti]=y0;v0z[ti]=z0;
    e1x[ti]=x1-x0;e1y[ti]=y1-y0;e1z[ti]=z1-z0;
    e2x[ti]=x2-x0;e2y[ti]=y2-y0;e2z[ti]=z2-z0;
    const lx=x0<x1?(x0<x2?x0:x2):(x1<x2?x1:x2);
    const ly=y0<y1?(y0<y2?y0:y2):(y1<y2?y1:y2);
    const lz=z0<z1?(z0<z2?z0:z2):(z1<z2?z1:z2);
    const hx=x0>x1?(x0>x2?x0:x2):(x1>x2?x1:x2);
    const hy=y0>y1?(y0>y2?y0:y2):(y1>y2?y1:y2);
    const hz=z0>z1?(z0>z2?z0:z2):(z1>z2?z1:z2);
    tLoX[ti]=lx;tLoY[ti]=ly;tLoZ[ti]=lz;
    tHiX[ti]=hx;tHiY[ti]=hy;tHiZ[ti]=hz;
    tCenX[ti]=(lx+hx)*0.5;tCenY[ti]=(ly+hy)*0.5;tCenZ[ti]=(lz+hz)*0.5;
    if(lx<gLoX)gLoX=lx;if(ly<gLoY)gLoY=ly;if(lz<gLoZ)gLoZ=lz;
    if(hx>gHiX)gHiX=hx;if(hy>gHiY)gHiY=hy;if(hz>gHiZ)gHiZ=hz;
    ti++;
   }
 }
 if(local){
  addGeometry(local.positions,3,local.indices,local.indices.length,identityMatrix);
 }else{
  const matrix=new T.Matrix4(),instance=new T.Matrix4();
  for(const object of objects){
   const g=object.geometry,pos=g.attributes.position,idx=g.index;
   const count=idx?idx.count:pos.count;
   for(let inst=0;inst<(object.isInstancedMesh?object.count:1);inst++){
    matrix.copy(object.matrixWorld);
    if(object.isInstancedMesh){object.getMatrixAt(inst,instance);matrix.multiply(instance);}
    addGeometry(pos.array,pos.itemSize,idx?idx.array:null,count,matrix.elements);
   }
  }
 }

 // Growable flat buffers for quantized RGBA32UI nodes & ordered triangle indices.
 let nodeCap=Math.max(256,triCount*4),nodeLen=0;
 let nodeData=new Uint32Array(nodeCap*4);
 let ordCap=Math.max(256,triCount*4),ordLen=0;
 let ordIdx=new Int32Array(ordCap);

 function ensureNodes(extra){
  if(nodeLen+extra<=nodeCap)return;
  while(nodeLen+extra>nodeCap)nodeCap*=2;
  const next=new Uint32Array(nodeCap*4);next.set(nodeData.subarray(0,nodeLen*4));nodeData=next;
 }
 function ensureOrd(extra){
  if(ordLen+extra<=ordCap)return;
  while(ordLen+extra>ordCap)ordCap*=2;
  const next=new Int32Array(ordCap);next.set(ordIdx.subarray(0,ordLen));ordIdx=next;
 }

 // Working arrays for per-cell clipped triangle bounds/centroids + indices.
 let workCap=Math.max(256,triCount);
 let wIdx=new Int32Array(workCap);
 let wLoX=new Float32Array(workCap),wLoY=new Float32Array(workCap),wLoZ=new Float32Array(workCap);
 let wHiX=new Float32Array(workCap),wHiY=new Float32Array(workCap),wHiZ=new Float32Array(workCap);
 let wCX=new Float32Array(workCap),wCY=new Float32Array(workCap),wCZ=new Float32Array(workCap);

 let qMinX=0,qMinY=0,qMinZ=0,qInvX=1,qInvY=1,qInvZ=1;

 function packBox(at,lx,ly,lz,hx,hy,hz,meta){
  const qlx=Math.max(0,Math.min(65535,Math.floor((lx-qMinX)*qInvX)));
  const qly=Math.max(0,Math.min(65535,Math.floor((ly-qMinY)*qInvY)));
  const qlz=Math.max(0,Math.min(65535,Math.floor((lz-qMinZ)*qInvZ)));
  const qhx=Math.max(qlx,Math.min(65535,Math.ceil((hx-qMinX)*qInvX)));
  const qhy=Math.max(qly,Math.min(65535,Math.ceil((hy-qMinY)*qInvY)));
  const qhz=Math.max(qlz,Math.min(65535,Math.ceil((hz-qMinZ)*qInvZ)));
  const o=at*4;
  nodeData[o]=(qlx|(qly<<16))>>>0;
  nodeData[o+1]=(qlz|(qhx<<16))>>>0;
  nodeData[o+2]=(qhy|(qhz<<16))>>>0;
  nodeData[o+3]=meta>>>0;
 }

 function partitionSAH(s,e,clx,cly,clz,chx,chy,chz){
  const extX=chx-clx,extY=chy-cly,extZ=chz-clz;
  let bestCost=Infinity,bestAx=-1,bestBin=0;
  for(let ax=0;ax<3;ax++){
   const cLo=ax===0?clx:(ax===1?cly:clz);
   const ext=ax===0?extX:(ax===1?extY:extZ);
   if(ext<=1e-7)continue;
   const sc=SAH_BINS/ext;
   const cArr=ax===0?wCX:(ax===1?wCY:wCZ);
   for(let b=0;b<SAH_BINS;b++){bCnt[b]=0;bLoX[b]=bLoY[b]=bLoZ[b]=Infinity;bHiX[b]=bHiY[b]=bHiZ[b]=-Infinity;}
   for(let i=s;i<e;i++){
    const k=wIdx[i];
    let bi=((cArr[k]-cLo)*sc)|0;
    if(bi>=SAH_BINS)bi=SAH_BINS-1;else if(bi<0)bi=0;
    bCnt[bi]++;
    if(wLoX[k]<bLoX[bi])bLoX[bi]=wLoX[k];if(wLoY[k]<bLoY[bi])bLoY[bi]=wLoY[k];if(wLoZ[k]<bLoZ[bi])bLoZ[bi]=wLoZ[k];
    if(wHiX[k]>bHiX[bi])bHiX[bi]=wHiX[k];if(wHiY[k]>bHiY[bi])bHiY[bi]=wHiY[k];if(wHiZ[k]>bHiZ[bi])bHiZ[bi]=wHiZ[k];
   }
   let rlx=Infinity,rly=Infinity,rlz=Infinity,rhx=-Infinity,rhy=-Infinity,rhz=-Infinity,rc=0;
   for(let b=SAH_BINS-1;b>0;b--){
    rc+=bCnt[b];
    if(bLoX[b]<rlx)rlx=bLoX[b];if(bLoY[b]<rly)rly=bLoY[b];if(bLoZ[b]<rlz)rlz=bLoZ[b];
    if(bHiX[b]>rhx)rhx=bHiX[b];if(bHiY[b]>rhy)rhy=bHiY[b];if(bHiZ[b]>rhz)rhz=bHiZ[b];
    rCnt[b]=rc;
    if(rc){const dx=rhx-rlx,dy=rhy-rly,dz=rhz-rlz;rArea[b]=dx*dy+dy*dz+dz*dx;}else rArea[b]=0;
   }
   let llx=Infinity,lly=Infinity,llz=Infinity,lhx=-Infinity,lhy=-Infinity,lhz=-Infinity,lc=0;
   for(let b=0;b<SAH_BINS-1;b++){
    lc+=bCnt[b];
    if(bLoX[b]<llx)llx=bLoX[b];if(bLoY[b]<lly)lly=bLoY[b];if(bLoZ[b]<llz)llz=bLoZ[b];
    if(bHiX[b]>lhx)lhx=bHiX[b];if(bHiY[b]>lhy)lhy=bHiY[b];if(bHiZ[b]>lhz)lhz=bHiZ[b];
    if(!lc||!rCnt[b+1])continue;
    const dx=lhx-llx,dy=lhy-lly,dz=lhz-llz;
    const cost=(dx*dy+dy*dz+dz*dx)*lc+rArea[b+1]*rCnt[b+1];
    if(cost<bestCost){bestCost=cost;bestAx=ax;bestBin=b+1;}
   }
  }
  let m=s;
  if(bestAx>=0){
   const cLo=bestAx===0?clx:(bestAx===1?cly:clz);
   const ext=bestAx===0?extX:(bestAx===1?extY:extZ);
   const sc=SAH_BINS/ext;
   const cArr=bestAx===0?wCX:(bestAx===1?wCY:wCZ);
   let i=s,j=e-1;
   while(i<=j){
    let bi=((cArr[wIdx[i]]-cLo)*sc)|0;if(bi>=SAH_BINS)bi=SAH_BINS-1;
    if(bi<bestBin){i++;}else{const tmp=wIdx[i];wIdx[i]=wIdx[j];wIdx[j]=tmp;j--;}
   }
   m=i;
  }
  if(m<=s||m>=e)m=(s+e)>>1;
  return m;
 }

 function buildChild(slot,s,e){
  let lx=Infinity,ly=Infinity,lz=Infinity,hx=-Infinity,hy=-Infinity,hz=-Infinity;
  let clx=Infinity,cly=Infinity,clz=Infinity,chx=-Infinity,chy=-Infinity,chz=-Infinity;
  for(let i=s;i<e;i++){
   const k=wIdx[i];
   const a=wLoX[k],b=wLoY[k],c=wLoZ[k],d=wHiX[k],f=wHiY[k],g=wHiZ[k];
   if(a<lx)lx=a;if(b<ly)ly=b;if(c<lz)lz=c;
   if(d>hx)hx=d;if(f>hy)hy=f;if(g>hz)hz=g;
   const cx=wCX[k],cy=wCY[k],cz=wCZ[k];
   if(cx<clx)clx=cx;if(cy<cly)cly=cy;if(cz<clz)clz=cz;
   if(cx>chx)chx=cx;if(cy>chy)chy=cy;if(cz>chz)chz=cz;
  }
  const n=e-s;
  if(n<=leafMax){
   ensureOrd(n);
   const first=ordLen;
   for(let i=s;i<e;i++)ordIdx[ordLen++]=wIdx[i];
   packBox(slot,lx,ly,lz,hx,hy,hz,(0x80000000|((first<<3)|n))>>>0);
   return;
  }
  const m=partitionSAH(s,e,clx,cly,clz,chx,chy,chz);
  ensureNodes(2);
  const pair=nodeLen;
  nodeLen+=2;
  packBox(slot,lx,ly,lz,hx,hy,hz,pair>>>0);
  buildChild(pair,s,m);
  buildChild(pair+1,m,e);
 }

 function buildTreeRoot(s,e){
  const n=e-s;
  if(n<=leafMax){
   ensureNodes(1);
   const at=nodeLen++;
   buildChild(at,s,e);
   return at;
  }
  let clx=Infinity,cly=Infinity,clz=Infinity,chx=-Infinity,chy=-Infinity,chz=-Infinity;
  for(let i=s;i<e;i++){
   const k=wIdx[i];
   const cx=wCX[k],cy=wCY[k],cz=wCZ[k];
   if(cx<clx)clx=cx;if(cy<cly)cly=cy;if(cz<clz)clz=cz;
   if(cx>chx)chx=cx;if(cy>chy)chy=cy;if(cz>chz)chz=cz;
  }
  const m=partitionSAH(s,e,clx,cly,clz,chx,chy,chz);
  ensureNodes(2);
  const pair=nodeLen;
  nodeLen+=2;
  buildChild(pair,s,m);
  buildChild(pair+1,m,e);
  return pair;
 }

 // Density-adaptive gridSize when gridSize === 'auto':
 // Keeps sparse scenes (e.g., Archive) and medium curved scenes (Atrium) at 1.0 m so flat leaf-ordered
 // triangles stay L2/SLC-resident, while gently refining ultra-dense 50k+ meshes down to >= 0.75 m.
 let resolvedGridSize=autoGrid?0:gridSize;
 if(autoGrid&&triCount>0){
  const dx=Math.max(gHiX-gLoX,1e-3),dy=Math.max(gHiY-gLoY,1e-3),dz=Math.max(gHiZ-gLoZ,1e-3);
  const vol=dx*dy*dz;
  const density=triCount/vol;
  let g=Math.cbrt(40.0/Math.max(density,1e-6));
  const minG=Math.max(0.75,range*2.5);
  g=Math.max(minG,Math.min(1.0,g));
  const maxDim=Math.max(dx,dy,dz);
  if(maxDim/g>max3DTextureSize-2)g=maxDim/(max3DTextureSize-2);
  if((dx/g)*(dy/g)*(dz/g)>65536)g=Math.cbrt(vol/65536);
  resolvedGridSize=Math.round(g*20)/20;
 }

 let gridData=new Uint32Array(2),gridMin=[0,0,0],gridDims=[1,1,1],maxTreeNodes=0;
 let rootMin=[0,0,0],rootSpan=[1,1,1];

 if(resolvedGridSize>0&&triCount>0){
  gridMin=[Math.floor(gLoX/resolvedGridSize)*resolvedGridSize,Math.floor(gLoY/resolvedGridSize)*resolvedGridSize,Math.floor(gLoZ/resolvedGridSize)*resolvedGridSize];
  gridDims=[Math.floor((gHiX-gridMin[0])/resolvedGridSize)+1,Math.floor((gHiY-gridMin[1])/resolvedGridSize)+1,Math.floor((gHiZ-gridMin[2])/resolvedGridSize)+1];
  if(gridDims[0]>max3DTextureSize||gridDims[1]>max3DTextureSize||gridDims[2]>max3DTextureSize)throw Error(`Contact grid exceeds device 3D texture limit ${max3DTextureSize}`);
  const gx=gridDims[0],gy=gridDims[1],gz=gridDims[2],nCells=gx*gy*gz;
  const invG=1/resolvedGridSize;

  // O(T) direct AABB-to-cell binning with exact triangle-plane vs expanded-cell separating-axis culling.
  const halfCell=0.5*resolvedGridSize+range;
  const counts=new Int32Array(nCells);
  for(let i=0;i<triCount;i++){
   const x0=Math.max(0,Math.floor((tLoX[i]-range-gridMin[0])*invG));
   const x1=Math.min(gx-1,Math.floor((tHiX[i]+range-gridMin[0])*invG));
   const y0=Math.max(0,Math.floor((tLoY[i]-range-gridMin[1])*invG));
   const y1=Math.min(gy-1,Math.floor((tHiY[i]+range-gridMin[1])*invG));
   const z0=Math.max(0,Math.floor((tLoZ[i]-range-gridMin[2])*invG));
   const z1=Math.min(gz-1,Math.floor((tHiZ[i]+range-gridMin[2])*invG));
   const ax=e1x[i],ay=e1y[i],az=e1z[i],bx=e2x[i],by=e2y[i],bz=e2z[i];
   const nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;
   const rPlane=halfCell*(Math.abs(nx)+Math.abs(ny)+Math.abs(nz))+1e-5;
   const vx=v0x[i],vy=v0y[i],vz=v0z[i];
   for(let z=z0;z<=z1;z++){
    const cz=gridMin[2]+(z+0.5)*resolvedGridSize-vz;
    for(let y=y0;y<=y1;y++){
     const cy=gridMin[1]+(y+0.5)*resolvedGridSize-vy;
     const yzDot=ny*cy+nz*cz;
     let off=(z*gy+y)*gx+x0;
     for(let x=x0;x<=x1;x++,off++){
      const cx=gridMin[0]+(x+0.5)*resolvedGridSize-vx;
      if(Math.abs(nx*cx+yzDot)<=rPlane)counts[off]++;
     }
    }
   }
  }
  const offsets=new Int32Array(nCells+1);
  for(let c=0;c<nCells;c++)offsets[c+1]=offsets[c]+counts[c];
  const cellTris=new Int32Array(offsets[nCells]);
  const cursor=new Int32Array(offsets.subarray(0,nCells));
  for(let i=0;i<triCount;i++){
   const x0=Math.max(0,Math.floor((tLoX[i]-range-gridMin[0])*invG));
   const x1=Math.min(gx-1,Math.floor((tHiX[i]+range-gridMin[0])*invG));
   const y0=Math.max(0,Math.floor((tLoY[i]-range-gridMin[1])*invG));
   const y1=Math.min(gy-1,Math.floor((tHiY[i]+range-gridMin[1])*invG));
   const z0=Math.max(0,Math.floor((tLoZ[i]-range-gridMin[2])*invG));
   const z1=Math.min(gz-1,Math.floor((tHiZ[i]+range-gridMin[2])*invG));
   const ax=e1x[i],ay=e1y[i],az=e1z[i],bx=e2x[i],by=e2y[i],bz=e2z[i];
   const nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;
   const rPlane=halfCell*(Math.abs(nx)+Math.abs(ny)+Math.abs(nz))+1e-5;
   const vx=v0x[i],vy=v0y[i],vz=v0z[i];
   for(let z=z0;z<=z1;z++){
    const cz=gridMin[2]+(z+0.5)*resolvedGridSize-vz;
    for(let y=y0;y<=y1;y++){
     const cy=gridMin[1]+(y+0.5)*resolvedGridSize-vy;
     const yzDot=ny*cy+nz*cz;
     let off=(z*gy+y)*gx+x0;
     for(let x=x0;x<=x1;x++,off++){
      const cx=gridMin[0]+(x+0.5)*resolvedGridSize-vx;
      if(Math.abs(nx*cx+yzDot)<=rPlane)cellTris[cursor[off]++]=i;
     }
    }
   }
  }

  gridData=new Uint32Array(nCells*2);
  const cellSpan=resolvedGridSize+2*range;
  const invSpan=65535/cellSpan;
  for(let z=0;z<gz;z++)for(let y=0;y<gy;y++)for(let x=0;x<gx;x++){
   const cid=(z*gy+y)*gx+x;
   const cStart=offsets[cid],cEnd=offsets[cid+1],cLen=cEnd-cStart;
   const first=nodeLen;
   if(cLen>0){
    const clx=gridMin[0]+x*resolvedGridSize-range,cly=gridMin[1]+y*resolvedGridSize-range,clz=gridMin[2]+z*resolvedGridSize-range;
    const chx=clx+cellSpan,chy=cly+cellSpan,chz=clz+cellSpan;
    qMinX=clx;qMinY=cly;qMinZ=clz;qInvX=invSpan;qInvY=invSpan;qInvZ=invSpan;
    for(let i=0;i<cLen;i++){
     const k=cellTris[cStart+i];
     wIdx[i]=k;
     // Clip triangle AABB to expanded cell bounds (100% exact for rays starting in cell of length <= range).
     const lx=tLoX[k]>clx?tLoX[k]:clx,ly=tLoY[k]>cly?tLoY[k]:cly,lz=tLoZ[k]>clz?tLoZ[k]:clz;
     const hx=tHiX[k]<chx?tHiX[k]:chx,hy=tHiY[k]<chy?tHiY[k]:chy,hz=tHiZ[k]<chz?tHiZ[k]:chz;
     wLoX[k]=lx;wLoY[k]=ly;wLoZ[k]=lz;wHiX[k]=hx;wHiY[k]=hy;wHiZ[k]=hz;
     wCX[k]=(lx+hx)*0.5;wCY[k]=(ly+hy)*0.5;wCZ[k]=(lz+hz)*0.5;
    }
    const root=buildTreeRoot(0,cLen);
    const cnt=nodeLen-first;
    if(cnt>maxTreeNodes)maxTreeNodes=cnt;
    gridData[cid*2]=root;gridData[cid*2+1]=cnt;
   }
  }
 }else if(triCount>0){
  const pad=1e-4;
  rootMin=[gLoX-pad,gLoY-pad,gLoZ-pad];
  rootSpan=[Math.max(gHiX-gLoX+2*pad,1e-3),Math.max(gHiY-gLoY+2*pad,1e-3),Math.max(gHiZ-gLoZ+2*pad,1e-3)];
  qMinX=rootMin[0];qMinY=rootMin[1];qMinZ=rootMin[2];
  qInvX=65535/rootSpan[0];qInvY=65535/rootSpan[1];qInvZ=65535/rootSpan[2];
  for(let i=0;i<triCount;i++){
   wIdx[i]=i;wLoX[i]=tLoX[i];wLoY[i]=tLoY[i];wLoZ[i]=tLoZ[i];
   wHiX[i]=tHiX[i];wHiY[i]=tHiY[i];wHiZ[i]=tHiZ[i];
   wCX[i]=tCenX[i];wCY[i]=tCenY[i];wCZ[i]=tCenZ[i];
  }
  buildTreeRoot(0,triCount);
  maxTreeNodes=nodeLen;
 }

 if(ordLen*8>=0x7FFFFFFF)throw Error('BVH triangle addressing exceeds 28-bit leaf range');
 if(local) return {nodes:nodeData.subarray(0,nodeLen*4),order:ordIdx.subarray(0,ordLen)};


 // Pack RGBA32UI node texture with fixed power-of-2 width (1024) for zero-div bit-shift addressing.
 const TEX_W=1024;
 const nWidth=TEX_W;
 const nHeight=Math.ceil(Math.max(1,nodeLen)/nWidth);
 if(nHeight>maxTextureSize)throw Error(`Contact BVH node texture exceeds device limit ${maxTextureSize}`);
 const nBuf=new Uint32Array(nWidth*nHeight*4);
 nBuf.set(nodeData.subarray(0,nodeLen*4));
 const nodeTexture=new T.DataTexture(nBuf,nWidth,nHeight,T.RGBAIntegerFormat,T.UnsignedIntType);
 nodeTexture.internalFormat='RGBA32UI';
 nodeTexture.minFilter=nodeTexture.magFilter=T.NearestFilter;
 nodeTexture.generateMipmaps=false;nodeTexture.needsUpdate=true;

 // Pack flat leaf-ordered RGBA32F triangles directly (zero contactTriIdx dependent texture fetch!).
 const tTexels=Math.max(1,ordLen*3);
 const tWidth=TEX_W;
 const tHeight=Math.ceil(tTexels/tWidth);
 if(tHeight>maxTextureSize)throw Error(`Contact BVH triangle texture exceeds device limit ${maxTextureSize}`);
 const tBuf=new Float32Array(tWidth*tHeight*4);
 for(let r=0;r<ordLen;r++){
  const k=ordIdx[r];
  const o=r*12;
  tBuf[o]=v0x[k];tBuf[o+1]=v0y[k];tBuf[o+2]=v0z[k];
  tBuf[o+4]=e1x[k];tBuf[o+5]=e1y[k];tBuf[o+6]=e1z[k];
  tBuf[o+8]=e2x[k];tBuf[o+9]=e2y[k];tBuf[o+10]=e2z[k];
 }
 const triangleTexture=new T.DataTexture(tBuf,tWidth,tHeight,T.RGBAFormat,T.FloatType);
 triangleTexture.minFilter=triangleTexture.magFilter=T.NearestFilter;
 triangleTexture.generateMipmaps=false;triangleTexture.needsUpdate=true;

 // Pack RG32UI 3D grid texture (8 B/cell instead of 16 B/cell!)
 const gridTexture=new T.Data3DTexture(gridData,...gridDims);
 gridTexture.format=T.RGIntegerFormat;gridTexture.type=T.UnsignedIntType;gridTexture.internalFormat='RG32UI';
 gridTexture.minFilter=gridTexture.magFilter=T.NearestFilter;gridTexture.unpackAlignment=1;gridTexture.needsUpdate=true;

 return {
  uniforms:{
   contactNodes:{value:nodeTexture},
   contactTriangles:{value:triangleTexture},
   contactNodeCount:{value:nodeLen},
   contactGrid:{value:gridTexture},
   contactGridMin:{value:new T.Vector3(...gridMin)},
   contactGridSize:{value:resolvedGridSize},
   contactRootMin:{value:new T.Vector3(...rootMin)},
   contactRootSpan:{value:new T.Vector3(...rootSpan)},
  },
  gridDims,gridSize:resolvedGridSize,maxTreeNodes,triangles:triCount,nodes:nodeLen,
  bytes:nBuf.byteLength+tBuf.byteLength+gridData.byteLength,
  dispose(){nodeTexture.dispose();triangleTexture.dispose();gridTexture.dispose();}
 };
}
