// // import { Toaster } from "@/components/ui/toaster";
// // import { Toaster as Sonner } from "@/components/ui/sonner";
// // import { TooltipProvider } from "@/components/ui/tooltip";
// // import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
// // import { BrowserRouter, Routes, Route } from "react-router-dom";
// // import { ThemeProvider } from "@/hooks/use-theme";
// // import Index from "./pages/Index";
// // import NotFound from "./pages/NotFound";
// // import MedicalImageAnalysis from "./pages/medical-image-analysis";
// // import PlaceHolder from "./pages/Placeholderpage";

// // // Create a placeholder component for routes that don't have pages yet
// // const PlaceholderPage = ({ title }: { title: string }) => (
// //   <div className="container mx-auto py-24 px-4">
// //     <h1 className="text-3xl font-bold mb-4">{title}</h1>
// //     <p>This page is under development. Please check back later.</p>
// //   </div>
// // );

// // const queryClient = new QueryClient();

// // const App = () => (
// //   <QueryClientProvider client={queryClient}>
// //     <ThemeProvider defaultTheme="light">
// //       <TooltipProvider>
// //         <Toaster />
// //         <Sonner />
// //         <BrowserRouter>
// //           <Routes>
// //             <Route path="/" element={<Index />} />
            
// //             {/* Feature routes from FeaturesSection */}
// //             <Route path="/medical-image-analysis" element={<MedicalImageAnalysis />} />
// //             <Route path="/find-doctor" element={<PlaceholderPage title="Find Nearby Doctors" />} />
// //             <Route path="/chatbot" element={<PlaceholderPage title="AI Chatbot Support" />} />
// //             <Route path="/reports" element={<PlaceholderPage title="Simplified Reports" />} />
// //             <Route path="/medical-records" element={<PlaceholderPage title="Secure Medical Records" />} />
// //             <Route path="/get-started" element={<PlaceholderPage title="Get Started" />} />
// //             <Route path="/PlaceHolderPage" element={<PlaceholderPage />} />
// //             {/* Catch-all route for 404 */}
// //             <Route path="*" element={<NotFound />} />
// //           </Routes>
// //         </BrowserRouter>
// //       </TooltipProvider>
// //     </ThemeProvider>
// //   </QueryClientProvider>
// // );

// // export default App;

// import { Toaster } from "@/components/ui/toaster";
// import { Toaster as Sonner } from "@/components/ui/sonner";
// import { TooltipProvider } from "@/components/ui/tooltip";
// import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
// import { BrowserRouter, Routes, Route } from "react-router-dom";
// import { ThemeProvider } from "@/hooks/use-theme";

// import Index from "./pages/Index";
// import NotFound from "./pages/NotFound";
// import MedicalImageAnalysis from "./pages/medical-image-analysis";
// import PlaceholderPage from "./pages/PlaceholderPage";

// const queryClient = new QueryClient();

// const App = () => (
//   <QueryClientProvider client={queryClient}>
//     <ThemeProvider defaultTheme="light">
//       <TooltipProvider>
//         <Toaster />
//         <Sonner />
//         <BrowserRouter>
//           <Routes>
//             <Route path="/" element={<Index />} />
            
//             {/* Feature routes from FeaturesSection */}
//             <Route path="/medical-image-analysis" element={<MedicalImageAnalysis />} />
//             <Route path="/PlaceholderPage" element={<PlaceholderPage title={""} />} />
//             <Route path="/chatbot" element={<PlaceholderPage title="AI Chatbot Support" />} />
//             <Route path="/reports" element={<PlaceholderPage title="Simplified Reports" />} />
//             <Route path="/medical-records" element={<PlaceholderPage title="Secure Medical Records" />} />
//             <Route path="/get-started" element={<PlaceholderPage title="Get Started" />} />
//             <Route path="/PlaceHolderPage" element={<PlaceholderPage title={""} />} />

//             {/* Catch-all route for 404 */}
//             <Route path="*" element={<NotFound />} />
//           </Routes>
//         </BrowserRouter>
//       </TooltipProvider>
//     </ThemeProvider>
//   </QueryClientProvider>
// );

// export default App;

import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@/hooks/use-theme";

import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import MedicalImageAnalysis from "./pages/medical-image-analysis";
import FindDoctor from "./pages/find-doctor"; // Import the FindDoctor page
import Record from "./pages/medical-records";
import Chatbot from "./pages/chatbot";
import StoreRecords from "./pages/Store-Record";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider defaultTheme="light">
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Index />} />

            {/* Feature routes */}
            <Route path="/medical-image-analysis" element={<MedicalImageAnalysis />} />
            <Route path="/find-doctor" element={<FindDoctor />} />
            <Route path="medical-records" element={<Record />} /> {/* Added find-doctor navigation */}
            <Route path="chatbot" element={<Chatbot />} />
            <Route path="/store-records" element={<StoreRecords />} />
            {/* <Route path="/chatbot" element={<PlaceholderPage title="AI Chatbot Support" />} />
            <Route path="/reports" element={<PlaceholderPage title="Simplified Reports" />} />
            <Route path="/medical-records" element={<PlaceholderPage title="Secure Medical Records" />} />
            <Route path="/get-started" element={<PlaceholderPage title="Get Started" />} /> */}

            {/* Catch-all route for 404 */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
