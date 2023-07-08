config_setting(
    name = "linux",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

configure_make( 
     name = "gsl", 
     lib_source = "//:gsl_sources", 
     out_include_dir = "include", 
     out_lib_dir = "lib", 
     out_static_libs = select({
         ":linux": ["libgsl.a", "libgslcblas.a"],
         ":macos": ["gsl.a", "gslcblas.a"],
         ":windows": ["gsl.lib", "gslcblas.lib"],
         "//conditions:default": [],
     }),
     visibility = ["//visibility:public"], 
 ) 

filegroup( 
     name = "gsl_sources", 
     srcs = glob(["**"]), 
     visibility = ["//visibility:public"], 
 )
