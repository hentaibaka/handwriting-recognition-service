export const useSendFile = () => {
  return {
    isLoading: false,
    sendFile: (file: File[]) => {
      console.log(file)
    },
  };
};
