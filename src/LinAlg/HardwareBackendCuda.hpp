#ifdef HIOP_USE_CUDA
namespace hiop
{


template<>
struct FeatureIsPresent<MemBackendCuda>
{
 static constexpr bool value = true;
};


} // end namespace hiop
#endif
