namespace hiop
{

template<>
struct FeatureIsPresent<MemBackendCpp>
{
 static constexpr bool value = true;
};

} // end namespace hiop
