namespace hiop
{

template<>
struct FeatureIsPresent<MemBackendCpp>
{
 static constexpr bool value = true;
};

template<>
struct SupportsHostMemSpace<MemBackendCpp>
{
  static constexpr bool value = true;
};


  
} // end namespace hiop
