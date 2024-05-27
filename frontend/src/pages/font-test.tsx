export const FontTest = () => {
  return (
    <div className="p-10">
      <h3 className="text-xl mb-5">Font Test</h3>

      <div className="font-involve flex flex-col px-10 text-3xl uppercase">
        <span className="font-normal">
          сервис преобразования рукописного текста в цифровой формат
        </span>
        <span className="font-medium">
          сервис преобразования рукописного текста в цифровой формат
        </span>
        <span className="font-semibold">
          сервис преобразования рукописного текста в цифровой формат
        </span>
        <span className="font-bold">
          сервис преобразования рукописного текста в цифровой формат
        </span>
      </div>

      <div className="mt-10 font-good-vibes flex flex-col px-10 text-3xl">
        <span className="font-normal">рукописного</span>
        <span className="font-normal">распознает</span>
      </div>

      <div className="mt-10 font-gilroy flex flex-col px-10 text-3xl">
        <span className="font-normal">Распознать свой файл (Jpeg, png, tiff)</span>
        <span className="font-medium">Распознать свой файл (Jpeg, png, tiff)</span>
        <span className="font-semibold">Распознать свой файл (Jpeg, png, tiff)</span>
      </div>
    </div>
  );
};
