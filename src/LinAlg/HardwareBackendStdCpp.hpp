namespace hiop
{

template<>
struct FeatureIsPresent<MemBackendCpp>
{
 static constexpr bool value = true;
};

template<class MemBackendCpp>
struct SupportsHostMemSpace
{
  static constexpr bool value = true;
};


  
} // end namespace hiop
